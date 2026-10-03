"""Isolated IDE adapter. One process per run keeps CLI globals and credentials private."""
import json
import os
import sys
from pathlib import Path, PurePosixPath

MAX_FILE = 1_000_000
BLOCKED = {'.git', '.skippy', '.env', 'node_modules', '.venv', '__pycache__'}


def safe_path(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Use a relative workspace path with forward slashes.')
    parts = PurePosixPath(value).parts
    if value.startswith('/') or any(p in ('..', '.') or p in BLOCKED or p.startswith('.env.') for p in parts):
        raise ValueError('Path is outside the editable workspace or is a private/generated file.')
    root = Path.cwd().resolve()
    result = (root / value).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Path escapes workspace.')
    return result


def main():
    output = sys.stdout
    sys.stdout = sys.stderr  # Rich/CLI diagnostics must not corrupt the event protocol.

    def emit(kind, **data):
        output.write(json.dumps({'type': kind, **data}, ensure_ascii=True) + '\n')
        output.flush()

    request = json.loads(sys.stdin.readline())
    from . import config, models
    config.configure(provider=request['provider'], model=request['model'], api_key=request.get('apiKey') or None)
    try:
        if request.get('action') == 'models':
            catalog = models.available_models()
            emit('models', models=list(catalog), selected=config.MODEL)
            return
        config.MODEL = models.validate_model(config.MODEL)
        models.apply_context_limit()
        from . import llm, tools, history
        from .todos import todos_prompt
        from .ui import ui

        baseline = {}
        for file in request.get('files', []):
            path = safe_path(file['path'])
            tools.write_atomic(path, file['content'])
            baseline[file['path']] = file['content']

        def approve(reason):
            emit('approval', reason=reason)
            reply = json.loads(sys.stdin.readline() or '{}')
            return reply.get('approved') is True

        ui.approve = approve
        allowed = {'read_file', 'write_file', 'str_replace', 'write_todos'}
        if os.getenv('SKIPPY_ALLOW_SHELL', '').lower() == 'true':
            allowed.add('bash')
        schemas = [s for s in tools.TOOL_SCHEMAS if s['function']['name'] in allowed]
        schemas += [{ 'type': 'function', 'function': {
            'name': 'list_files', 'description': 'List all editable workspace files.',
            'parameters': {'type': 'object', 'properties': {}}}}]

        def snapshot():
            result = {}
            for path in Path.cwd().rglob('*'):
                relative = path.relative_to(Path.cwd()).as_posix()
                try:
                    safe_path(relative)
                    if path.is_file() and path.stat().st_size <= MAX_FILE:
                        result[relative] = path.read_text(encoding='utf-8')
                except (ValueError, OSError, UnicodeError):
                    continue
            return result

        def changes():
            nonlocal baseline
            current = snapshot()
            for path, content in current.items():
                if baseline.get(path) != content:
                    emit('file', path=path, content=content, previous=baseline.get(path))
            for path in baseline.keys() - current.keys():
                emit('error', message=f'{path} was deleted by a command. Deletion was not applied to the IDE; remove it manually if intended.')
            baseline = current

        system = ('You are Skippy, the coding harness inside a collaborative IDE. '
                  'Use tools to actually inspect, create and edit workspace files, then verify your changes. '
                  'Use relative paths. Do not claim edits or tests that did not happen. '
                  'Use write_todos for multi-step tasks. Do not read secrets or generated directories. '
                  'The workspace is a per-run copy of the browser files; GitHub push is a separate user action. '
                  'Shell commands need explicit approval and run on the host machine. '
                  + ('Shell is PowerShell. No && or POSIX commands. ' if os.name == 'nt' else 'Shell is POSIX. ')
                  + ('Shell execution is disabled; explain when a test cannot be run. ' if 'bash' not in allowed else '')
                  + '\nWorkspace files:\n' + '\n'.join(baseline))
        messages = [{'role': 'system', 'content': system}]
        messages.extend(request.get('history', [])[-12:])
        messages.append({'role': 'user', 'content': request['prompt']})
        failures = {}
        for step in range(min(config.positive_int('MAX_AGENT_STEPS', 20), 50)):
            history.fit(messages)
            plan = todos_prompt()
            prepared = messages + ([{'role': 'system', 'content': 'Current plan:\n' + plan}] if plan else [])
            emit('status', message=f'Thinking · step {step + 1}')
            message, usage = llm.call_llm(prepared, tools=schemas)
            messages.append(message.model_dump(exclude_none=True))
            if message.content:
                emit('message', content=message.content)
            if not message.tool_calls:
                if not message.content:
                    raise ValueError('Model returned an empty answer. Try another model.')
                emit('done', usage=usage)
                return
            for call in message.tool_calls:
                name = call.function.name
                emit('tool', name=name, arguments=call.function.arguments, status='running', id=call.id)
                try:
                    args = json.loads(call.function.arguments)
                    if not isinstance(args, dict):
                        raise ValueError('Tool arguments must be an object.')
                    if name == 'list_files':
                        result = '\n'.join(snapshot()) or '(empty workspace)'
                    elif name not in allowed:
                        result = 'Error: this tool is not available in the IDE.'
                    else:
                        if name in {'read_file', 'write_file', 'str_replace'}:
                            safe_path(args.get('path'))
                        if len(str(args.get('content', args.get('new_str', '')))) > MAX_FILE:
                            raise ValueError('File exceeds the 1 MB limit.')
                        if name == 'bash' and not approve('Run on the server: ' + str(args.get('command', ''))):
                            result = 'The user denied this tool call.'
                        elif name == 'bash':
                            # Explicit approval is already recorded; retain command validation and deny rules.
                            from .permissions import check
                            action, reason = check(name, args)
                            error = tools.sandbox.command_error(args['command'])
                            result = 'Error: ' + (error or reason) if error or action == 'deny' else tools.bash(**args)
                        else:
                            _, result = tools.execute(call, allowed_names=allowed)
                except Exception as exc:
                    result = 'Error: ' + str(exc)
                result = str(result)
                emit('tool', name=name, result=result[:16000], status='failed' if tools.failed_result(result) else 'success', id=call.id)
                messages.append({'role': 'tool', 'tool_call_id': call.id, 'content': result})
                changes()
                if result == 'The user denied this tool call.':
                    emit('done', message='Stopped after command was denied.')
                    return
                key = name + call.function.arguments
                if tools.failed_result(result):
                    failures[key] = failures.get(key, 0) + 1
                    if failures[key] >= 3:
                        raise ValueError('Stopped after the same tool failed three times. Review its error and retry.')
        raise ValueError('Agent reached the step limit. Send a follow-up to continue.')
    except Exception as exc:
        detail = str(exc)
        if config.API_KEY:
            detail = detail.replace(config.API_KEY, '[redacted]')
        emit('error', message=detail)


if __name__ == '__main__':
    main()
