import json
import os
import tempfile
import io
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from openai.types.chat import ChatCompletionMessage

from skippy_harness import agent, commands, compact, config, history, llm, models
from skippy_harness import permissions, sandbox, session, skills, subagent, todos
from skippy_harness import tools, prompt, context
from rich.console import Console
from prompt_toolkit.document import Document
from skippy_harness.ui import UI


def call(name, args, call_id="test-call"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def assistant_call(name, args, call_id="test-call"):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        for obj, name, value in [
            (config, "ACTIVE_PROVIDER", ""), (config, "MODEL", ""),
            (config, "BASE_URL", ""), (config, "API_KEY", ""),
            (config, "CONTEXT_WINDOW", 128000), (config, "_SELECTED_MODELS", {}),
            (config, "PROJECT_ENV", self.root / ".env"),
            (models, "_CACHE", {}), (llm, "_client", None),
            (session, "SESSION_DIR", self.root / "sessions"),
            (session, "LEGACY_DIR", self.root / "legacy"),
            (session, "CURRENT", "test"), (session, "WRITTEN", 0),
            (session, "BASE_TODOS", []),
            (todos, "TODOS", []),
        ]:
            p = patch.object(obj, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_provider_switch_keeps_separate_models_and_keys(self):
        os.environ.update(PROVIDER="openrouter", MODEL="vendor/cloud", OPENROUTER_API_KEY="test-key")
        config.configure(provider="openrouter")
        self.assertEqual(config.MODEL, "vendor/cloud")
        config.configure(provider="ollama", model="local:2b")
        self.assertEqual(config.API_KEY, "ollama")
        config.configure(provider="openrouter")
        self.assertEqual(config.MODEL, "vendor/cloud")
        self.assertEqual(config.API_KEY, "test-key")
        config.configure(provider="ollama")
        self.assertEqual(config.MODEL, "local:2b")

    def test_active_provider_readiness_does_not_use_other_keys(self):
        os.environ.update(PROVIDER="nvidia", NVIDIA_API_KEY="nvidia-key")
        config.configure(provider="openrouter")
        self.assertFalse(config.is_configured())
        config.configure(provider="ollama")
        self.assertTrue(config.is_configured())

    def test_saved_model_and_ollama_url_survive_restart(self):
        os.environ["OLLAMA_HOST"] = "localhost:12345"
        config.configure(provider="ollama", model="local:latest")
        config.persist_selection()
        self.assertEqual(config.BASE_URL, "http://localhost:12345/v1")
        config._SELECTED_MODELS.clear()
        config.ACTIVE_PROVIDER = config.MODEL = ""
        config.configure()
        self.assertEqual(config.MODEL, "local:latest")
        self.assertEqual(config.BASE_URL, "http://localhost:12345/v1")
        self.assertIn("OLLAMA_MODEL=local:latest", config.PROJECT_ENV.read_text())

    def test_configure_invalidates_client(self):
        llm._client = object()
        config.configure(provider="ollama")
        self.assertIsNone(llm._client)

    def test_bad_integer_setting_does_not_crash(self):
        os.environ["CONTEXT_WINDOW"] = "invalid"
        config.configure(provider="ollama")
        self.assertEqual(config.CONTEXT_WINDOW, 8192)

    def test_unknown_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown provider"):
            config.configure(provider="olama")

    def test_ollama_discovery_lists_installed_models(self):
        config.configure(provider="ollama")
        with patch.object(models, "request_json", return_value={"models": [{"name": "actual:2b"}]}):
            self.assertEqual(list(models.available_models()), ["actual:2b"])

    def test_ollama_missing_models_has_actionable_error(self):
        config.configure(provider="ollama")
        with patch.object(models, "request_json", return_value={"models": []}):
            with self.assertRaisesRegex(ValueError, "ollama pull"):
                models.available_models()

    def test_ollama_tool_capability_is_required(self):
        config.configure(provider="ollama")
        with patch.object(models, "request_json", side_effect=[{"models": [{"name": "plain:latest"}]}, {"capabilities": ["completion"]}]):
            with self.assertRaisesRegex(ValueError, "does not support tools"):
                models.validate_model("plain")

    def test_openrouter_only_lists_text_models_with_tools(self):
        config.configure(provider="openrouter")
        data = {"data": [
            {"id": "good", "supported_parameters": ["tools"], "context_length": 32000},
            {"id": "chat-only", "supported_parameters": []},
            {"id": "image", "supported_parameters": ["tools"], "architecture": {"output_modalities": ["image"]}},
        ]}
        with patch.object(models, "request_json", return_value=data):
            self.assertEqual(list(models.available_models()), ["good"])
            config.MODEL = "good"
            models.apply_context_limit()
            self.assertEqual(config.CONTEXT_WINDOW, 32000)

    def test_invalid_selection_preserves_current_model(self):
        config.configure(provider="ollama", model="working:2b")
        with patch.object(models, "validate_model", side_effect=ValueError("missing")), patch.object(commands, "ui"):
            commands.switch_model("/model nonexistent", [])
        self.assertEqual(config.MODEL, "working:2b")

    def test_model_command_persists_selection(self):
        config.configure(provider="ollama", model="old")
        with patch.object(models, "validate_model", return_value="new:2b"), patch.object(commands, "redraw", side_effect=lambda m, _:m), patch.object(commands, "ui"):
            commands.switch_model("/model new:2b", [])
        self.assertEqual(config.MODEL, "new:2b")
        self.assertIn("OLLAMA_MODEL=new:2b", config.PROJECT_ENV.read_text())

    def test_direct_provider_command_selects_and_persists_correct_backend(self):
        config.configure(provider="openrouter", model="cloud/model")
        with patch.object(models, "validate_model", return_value="local:2b"), \
             patch.object(commands, "ui"), patch.object(commands, "redraw", side_effect=lambda m, _:m):
            commands.handle("/provider ollama", [])
        self.assertEqual(config.ACTIVE_PROVIDER, "ollama")
        self.assertEqual(config.MODEL, "local:2b")
        self.assertIn("PROVIDER=ollama", config.PROJECT_ENV.read_text())

    def test_menu_uses_one_based_selection(self):
        ui = UI()
        with patch.object(ui.console, "print"), patch("skippy_harness.ui.prompt.read", return_value="1"):
            self.assertEqual(ui.pick("pick", ["first", "second"]), 0)

    @unittest.skipUnless(os.name == "nt", "Windows output encoding")
    def test_windows_tool_output_supports_unicode(self):
        buffer = io.BytesIO()
        output = io.TextIOWrapper(buffer, encoding="cp1252")
        with patch("sys.stdout", output):
            ui = UI()
            ui.tool("read_file", {"path":"sample"}, "Unicode: \u4e16\u754c")
        output.flush()
        self.assertIn("\u4e16\u754c".encode("utf-8"), buffer.getvalue())

    def test_menu_search_and_invalid_zero(self):
        ui = UI()
        with patch.object(ui.console, "print"), patch("skippy_harness.ui.prompt.read", side_effect=["0", "second", "2"]):
            self.assertEqual(ui.pick("pick", ["first", "second"]), 1)

    def test_expanding_tools_keeps_output_after_history_is_trimmed(self):
        output = io.StringIO()
        ui = UI()
        ui.console = Console(file=output, width=80, color_system=None)
        result = "\n".join(f"line {i}" for i in range(40))
        ui.tool("read_file", {"path":"sample"}, result, call_id="cached-call")
        self.assertNotIn("line 39", output.getvalue())
        output.seek(0)
        output.truncate(0)
        ui.toggle_tools()
        ui.replay([assistant_call("read_file", {"path":"sample"}, "cached-call"),
                   {"role":"tool", "tool_call_id":"cached-call", "content":"[output trimmed]"}])
        self.assertIn("line 39", output.getvalue())

    def test_compact_shell_failure_remains_visible(self):
        output = io.StringIO()
        ui = UI()
        ui.console = Console(file=output, width=36, color_system=None)
        ui.tool("bash", {"command":"python -m unittest discover -s tests -v"}, "Exit code: 7\nAssertionError: failed")
        rendered = output.getvalue()
        self.assertIn("[exit 7]", rendered)
        self.assertIn("AssertionError", rendered)
        self.assertNotIn("Exit code:", rendered)

    def test_tool_toggle_preserves_input_and_cursor_through_other_prompts(self):
        draft = Document("finish this task", cursor_position=6)
        event = SimpleNamespace(current_buffer=SimpleNamespace(document=draft), app=MagicMock())
        fake_session = MagicMock()
        fake_session.prompt.return_value = "hello"
        with patch.object(prompt, "DRAFT", None), patch.object(prompt, "SESSION", fake_session):
            prompt._toggle_tools(event)
            event.app.exit.assert_called_once_with(result="/tools")
            prompt.read("allow? ")
            self.assertEqual(prompt.DRAFT, draft)
            prompt.read("> ")
            self.assertEqual(fake_session.prompt.call_args.kwargs["default"], draft)
            self.assertIsNone(prompt.DRAFT)

    def test_tool_display_command_does_not_change_transcript(self):
        messages = [{"role":"system", "content":"sys"}, {"role":"user", "content":"work"}]
        with patch.object(commands, "ui") as display, patch.object(commands, "redraw", side_effect=lambda m, _:m):
            self.assertIs(commands.handle("/tools", messages), messages)
        display.toggle_tools.assert_called_once()

    def test_empty_tool_list_really_disables_tools(self):
        config.configure(provider="openrouter")
        fake = MagicMock()
        fake.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=ChatCompletionMessage(role="assistant", content="ok"))], usage=None)
        with patch.object(llm, "get_client", return_value=fake):
            llm.call_llm([{"role": "user", "content": "hello"}], tools=[])
        self.assertNotIn("tools", fake.chat.completions.create.call_args.kwargs)

    def test_native_ollama_tool_call_conversion(self):
        config.configure(provider="ollama", model="test")
        response = {"message": {"content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "sample"}}}]}, "prompt_eval_count": 12, "eval_count": 5}
        with patch.object(models, "request_json", return_value=response) as request:
            message, usage = llm.call_llm([{"role": "user", "content": "inspect"}])
        self.assertTrue(message.tool_calls[0].id.startswith("call_"))
        self.assertEqual(json.loads(message.tool_calls[0].function.arguments), {"path": "sample"})
        self.assertEqual(usage["prompt_tokens"], 12)
        self.assertEqual(request.call_args.kwargs["json_body"]["options"]["num_ctx"], 8192)

    def test_native_ollama_receives_tool_name_and_decoded_arguments(self):
        transcript = [assistant_call("read_file", {"path":"sample"}), {"role":"tool", "tool_call_id":"test-call", "content":"contents"}]
        converted = llm.ollama_messages(transcript)
        self.assertEqual(converted[0]["tool_calls"][0]["function"]["arguments"], {"path":"sample"})
        self.assertEqual(converted[1]["tool_name"], "read_file")

    def test_native_ollama_survives_previous_invalid_arguments(self):
        message = assistant_call("read_file", {})
        message["tool_calls"][0]["function"]["arguments"] = "{broken"
        converted = llm.ollama_messages([message, {"role":"tool", "tool_call_id":"test-call", "content":"Error: invalid JSON"}])
        self.assertEqual(converted[0]["tool_calls"][0]["function"]["arguments"], {})

    def test_openrouter_monthly_limit_error_explains_local_fallback(self):
        config.configure(provider="openrouter")
        fake = MagicMock()
        fake.chat.completions.create.side_effect = RuntimeError("Key limit exceeded (monthly limit)")
        with patch.object(llm, "get_client", return_value=fake):
            with self.assertRaisesRegex(ValueError, "/provider ollama"):
                llm.call_llm([{"role":"user", "content":"hello"}])

    def test_compaction_uses_active_backend_without_tools(self):
        with patch.object(compact, "call_llm", return_value=(SimpleNamespace(content="summary"), {})) as llm_call:
            self.assertEqual(compact.summarize([{"role":"user", "content":"work"}]), "summary")
        self.assertEqual(llm_call.call_args.kwargs["tools"], [])

    def test_shell_preserves_exit_status(self):
        with patch.object(sandbox, "run", return_value=SimpleNamespace(stdout="", stderr="", returncode=7)):
            self.assertIn("Exit code: 7", tools.bash("some command"))

    def test_files_support_unicode_and_missing_parents(self):
        path = self.root / "nested" / "unicode.txt"
        tools.write_file(str(path), "hello \u4e16\u754c")
        self.assertEqual(tools.read_file(str(path)), "hello \u4e16\u754c")
        tools.str_replace(str(path), "hello", "welcome")
        self.assertEqual(tools.read_file(str(path)), "welcome \u4e16\u754c")
        self.assertTrue(tools.str_replace(str(path), "", "oops").startswith("Error:"))

    def test_subagents_cannot_write_even_if_model_invents_call(self):
        names = {s["function"]["name"] for s in subagent.toolset()}
        self.assertNotIn("write_file", names)
        path = self.root / "forbidden"
        _, result = tools.execute(call("write_file", {"path": str(path), "content": "oops"}), allowed_names=names, read_only=True)
        self.assertIn("Blocked", result)
        self.assertFalse(path.exists())

    def test_subagents_cannot_run_unapproved_shell(self):
        with patch.object(sandbox, "run") as run:
            _, result = tools.execute(call("bash", {"command":"python script.py"}), read_only=True)
        self.assertIn("Blocked", result)
        run.assert_not_called()

    def test_permission_bypasses_require_approval(self):
        for command in ["echo hello > file", "echo $(rm file)", "git status\nrm file", "find . -delete", "rg --pre=script foo", "git diff --output=file"]:
            with self.subTest(command=command):
                self.assertNotEqual(permissions.decide(command), "allow")
        self.assertEqual(permissions.decide('rg "cap|max" .'), "allow")
        self.assertEqual(permissions.decide("git push"), "deny")

    def test_non_object_arguments_are_handled(self):
        self.assertIn("JSON object", tools.execute(call("read_file", []))[1])

    def test_wrong_argument_types_do_not_truncate_existing_file(self):
        target = self.root / "original.txt"
        target.write_text("keep me", encoding="utf-8")
        _, result = tools.execute(call("write_file", {"path":str(target), "content":5}))
        self.assertIn("must be a string", result)
        self.assertEqual(target.read_text(), "keep me")

    def test_failed_atomic_write_preserves_original(self):
        target = self.root / "original.txt"
        target.write_text("keep me", encoding="utf-8")
        with self.assertRaises(UnicodeEncodeError):
            tools.write_file(str(target), "invalid surrogate \ud800")
        self.assertEqual(target.read_text(), "keep me")

    def test_invalid_todos_do_not_corrupt_state(self):
        valid = [{"content":"work", "activeForm":"working", "status":"in_progress"}]
        todos.write_todos(valid)
        self.assertTrue(todos.write_todos([{"status":"oops"}]).startswith("Error:"))
        self.assertEqual(todos.TODOS, valid)

    def test_malformed_skill_does_not_crash_discovery(self):
        directory = self.root / "skills" / "broken"
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text("no frontmatter")
        with patch.object(skills, "SKILL_DIRS", [directory.parent]), self.assertWarns(UserWarning):
            self.assertEqual(skills.find_skills(), {})

    def test_fit_can_drop_already_capped_outputs(self):
        config.CONTEXT_WINDOW = 200
        messages = [{"role":"system", "content":"hi"}, {"role":"tool", "content": "x"*3000 + history.TRIMMED}]
        self.assertEqual(history.fit(messages), 1)
        self.assertLess(history.estimate(messages), config.CONTEXT_WINDOW * config.COMPACT_AT)

    def test_finished_turn_strips_spill_pointer(self):
        messages = [{"role":"tool", "content":"x"*10000 + history.TRIMMED + " /tmp/gone"}]
        history.strip(messages)
        self.assertNotIn("/tmp/gone", messages[0]["content"])
        once = messages[0]["content"]
        history.strip(messages)
        self.assertEqual(messages[0]["content"], once)

    def test_session_rewind_compaction_and_plan_restore(self):
        messages = [{"role":"system", "content":"test"}, {"role":"user", "content":"hello"}]
        session.save(messages)
        self.assertEqual(session.load("test"), messages)
        todos.write_todos([{"content":"work", "activeForm":"working", "status":"in_progress"}])
        session.compacted(messages)
        todos.TODOS.clear()
        session.open_session("test")
        self.assertEqual(todos.active_form(), "working")
        session.rewind_to(1)
        self.assertEqual(len(session.load("test")), 1)

    def test_session_resumes_interrupted_tool_batch_and_persists_repair(self):
        messages = [{"role":"system", "content":"test"}, assistant_call("read_file", {"path":"file"})]
        session.save(messages)
        restored = session.open_session("test")
        self.assertEqual(restored[-1]["role"], "tool")
        self.assertEqual(restored[-1]["tool_call_id"], "test-call")
        self.assertEqual(session.load("test", repair_history=False), restored)
        session.save(restored + [{"role":"user", "content":"continue"}])
        self.assertEqual(len(session.load("test")), 4)

    def test_session_ignores_corrupt_lines_and_non_messages(self):
        session.SESSION_DIR.mkdir()
        session.path_for("test").write_text('[]\n{}\n{broken\n'+json.dumps({"role":"user", "content":"valid"})+'\n', encoding="utf-8")
        self.assertEqual(session.load("test"), [{"role":"user", "content":"valid"}])

    def test_legacy_session_import_preserves_original(self):
        session.LEGACY_DIR.mkdir()
        legacy = session.LEGACY_DIR / "old.jsonl"
        legacy.write_text(json.dumps({"role":"user", "content":"old"})+"\n", encoding="utf-8")
        self.assertEqual(session.all_sessions()[0]["id"], "old")
        self.assertEqual(session.open_session("old")[0]["content"], "old")
        self.assertTrue(legacy.exists())
        self.assertTrue(session.path_for("old").exists())

    def test_rewind_only_cuts_complete_user_turns(self):
        messages = [{"role":"system", "content":"sys"}, {"role":"user", "content":"one"},
                    assistant_call("read_file", {"path":"file"}), {"role":"tool", "tool_call_id":"test-call", "content":"ok"},
                    {"role":"assistant", "content":"done"}, {"role":"user", "content":"two"}]
        session.save(messages)
        with patch.object(commands, "redraw", side_effect=lambda m, _:m):
            restored = commands.rewind(messages, 1)
        self.assertEqual(restored[-1]["content"], "done")
        self.assertEqual(len(restored), 5)

    def test_replay_survives_malformed_saved_tool_arguments(self):
        ui = UI()
        message = assistant_call("read_file", {})
        message["tool_calls"][0]["function"]["arguments"] = "{broken"
        with patch.object(ui, "tool") as tool:
            ui.replay([message])
        tool.assert_called_once()

    def test_cli_ollama_override_skips_wrong_provider_setup(self):
        os.environ.update(PROVIDER="openrouter")
        fake_ui = MagicMock()
        fake_ui.ask.return_value = None
        with patch("sys.argv", ["skippy", "--provider", "ollama", "--model", "local:2b"]), \
             patch.object(agent, "ui", fake_ui), patch.object(config, "interactive_setup") as setup, \
             patch.object(models, "validate_model", return_value="local:2b"):
            agent.main()
        setup.assert_not_called()
        self.assertEqual(config.ACTIVE_PROVIDER, "ollama")

    def test_main_loop_executes_tool_result_before_next_request(self):
        config.configure(provider="ollama", model="local:2b")
        fake_ui = MagicMock()
        fake_ui.ask.side_effect = ["write a file", None]
        fake_ui.working.side_effect = lambda _: nullcontext()
        target = self.root / "agent.txt"
        first = ChatCompletionMessage.model_validate(assistant_call("write_file", {"path":str(target), "content":"verified"}))
        second = ChatCompletionMessage(role="assistant", content="Done")
        seen = []
        def llm_call(messages):
            seen.append(list(messages))
            return (first if len(seen)==1 else second), {"prompt_tokens":100}
        with patch("sys.argv", ["skippy"]), patch.object(agent, "ui", fake_ui), \
             patch.object(models, "validate_model", return_value="local:2b"), \
             patch.object(agent, "call_llm", side_effect=llm_call), \
             patch.object(agent, "reminder", return_value={"role":"user", "content":"env"}):
            agent.main()
        self.assertEqual(target.read_text(), "verified")
        self.assertEqual(seen[1][-1]["role"], "tool")
        self.assertIn("Wrote", seen[1][-1]["content"])

    def test_machine_context_does_not_replace_user_request_or_mutate_history(self):
        messages = [{"role":"system", "content":"instructions"}, {"role":"user", "content":"hello"}]
        prepared = context.prepare_messages(messages, {"content":"branch: main"})
        self.assertEqual(prepared[-1], {"role":"user", "content":"hello"})
        self.assertIn("branch: main", prepared[0]["content"])
        self.assertEqual(messages[0]["content"], "instructions")
        self.assertEqual(sum(m["role"] == "user" for m in prepared), 1)

    def test_old_session_uses_current_system_prompt(self):
        messages = [{"role":"system", "content":"Always code, even for hello"}, {"role":"user", "content":"hello"}]
        refreshed = llm.refresh_system_prompt(messages)
        self.assertEqual(refreshed[0]["content"], llm.SYSTEM_PROMPT)
        self.assertEqual(refreshed[1:], messages[1:])
        self.assertEqual(messages[0]["content"], "Always code, even for hello")

    def test_invalid_powershell_commands_fail_before_permission_prompt(self):
        with patch.object(sandbox.sys, "platform", "win32"), patch.object(sandbox.shutil, "which", return_value=None), patch("skippy_harness.ui.ui.approve") as approve:
            for command in ["cd project && dir /B", "ls -la", "dir /B", "which git"]:
                with self.subTest(command=command):
                    result = tools.execute(call("bash", {"command":command}))[1]
                    self.assertTrue(result.startswith("Error:"))
            approve.assert_not_called()
            self.assertIsNone(sandbox.command_error('Write-Output "example && quoted"'))
            self.assertIsNone(sandbox.command_error("Get-ChildItem -Name"))

    def test_readonly_windows_aliases_do_not_require_approval(self):
        for command in ["dir", "dir *", "Get-ChildItem -Name", "Get-Command git"]:
            self.assertEqual(permissions.decide(command), "allow")

    def test_denied_tool_stops_turn_and_skips_rest_of_batch(self):
        config.configure(provider="ollama", model="local:2b")
        fake_ui = MagicMock()
        fake_ui.ask.side_effect = ["run two commands", None]
        fake_ui.working.side_effect = lambda _: nullcontext()
        message = assistant_call("bash", {"command":"python script.py"}, "first")
        message["tool_calls"] += assistant_call("write_file", {"path":"unwanted.txt", "content":"oops"}, "second")["tool_calls"]
        first = ChatCompletionMessage.model_validate(message)
        with patch("sys.argv", ["skippy"]), patch.object(agent, "ui", fake_ui), \
             patch.object(models, "validate_model", return_value="local:2b"), \
             patch.object(agent, "call_llm", return_value=(first, {"prompt_tokens":100})) as llm_call, \
             patch.object(agent, "execute", return_value=({}, "The user denied this tool call.")) as execute, \
             patch.object(agent, "reminder", return_value={"content":"env"}):
            agent.main()
        llm_call.assert_called_once()
        execute.assert_called_once()
        results = [m for m in session.load("test") if m["role"] == "tool"]
        self.assertEqual(len(results), 2)
        self.assertTrue(results[-1]["content"].startswith("Skipped:"))


if __name__ == "__main__":
    unittest.main()
