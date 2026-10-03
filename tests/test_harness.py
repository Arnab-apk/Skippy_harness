import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from openai.types.chat import ChatCompletionMessage

from skippy_harness import agent, commands, compact, config, history, llm, models
from skippy_harness import permissions, sandbox, session, skills, subagent, todos
from skippy_harness import tools
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

    def test_menu_uses_one_based_selection(self):
        ui = UI()
        with patch.object(ui.console, "print"), patch("skippy_harness.ui.prompt.read", return_value="1"):
            self.assertEqual(ui.pick("pick", ["first", "second"]), 0)

    def test_menu_search_and_invalid_zero(self):
        ui = UI()
        with patch.object(ui.console, "print"), patch("skippy_harness.ui.prompt.read", side_effect=["0", "second", "2"]):
            self.assertEqual(ui.pick("pick", ["first", "second"]), 1)

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
        self.assertEqual(seen[1][-2]["role"], "tool")
        self.assertIn("Wrote", seen[1][-2]["content"])


if __name__ == "__main__":
    unittest.main()
