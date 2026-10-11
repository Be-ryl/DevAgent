"""Tests for the runtime-bound tool registry."""

import json
from pathlib import Path
import tempfile
import unittest

from pydantic import BaseModel

from devagent.core.state import RuntimeState
from devagent.tools.registry import (
    InvalidToolArgumentsError,
    ToolRegistry,
    UnknownToolError,
    build_tools,
)


class RecordingArgs(BaseModel):
    """Arguments for the local recording tool used by registry tests."""

    value: int


class ToolRegistryTest(unittest.TestCase):
    """Tests for registered tool lookup and invocation."""

    def test_registered_write_file_writes_without_explicit_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")
            tools = build_tools(state)

            write_tool = tools["write_file"]
            result = write_tool["function"]("notes/result.txt", "created\n")

            self.assertEqual(
                set(tools),
                {
                    "read_file",
                    "write_file",
                    "edit_file",
                    "search_text",
                    "run_command",
                },
            )
            self.assertEqual(write_tool["name"], "write_file")
            self.assertTrue(write_tool["description"])
            self.assertTrue(result["ok"])
            self.assertEqual(result["path"], "notes/result.txt")
            self.assertEqual(
                (state.workspace / "notes" / "result.txt").read_text(
                    encoding="utf-8"
                ),
                "created\n",
            )

    def test_openai_schema_contains_tool_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")
            registry = build_tools(state)

            schema = next(
                item
                for item in registry.openai_schemas()
                if item["function"]["name"] == "write_file"
            )

            self.assertEqual(schema["type"], "function")
            self.assertEqual(schema["function"]["name"], "write_file")
            self.assertEqual(
                schema["function"]["description"],
                registry["write_file"]["description"],
            )
            parameters = schema["function"]["parameters"]
            self.assertEqual(parameters["type"], "object")
            self.assertEqual(set(parameters["properties"]), {"file_path", "content"})
            self.assertEqual(set(parameters["required"]), {"file_path", "content"})

    def test_all_registered_tools_export_json_serializable_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")
            registry = build_tools(state)

            schemas = registry.openai_schemas()

            self.assertEqual(
                {schema["function"]["name"] for schema in schemas},
                set(registry),
            )
            json.dumps(schemas)


class ToolInvocationTest(unittest.TestCase):
    """Tests for parsing and validating arguments before execution."""

    def setUp(self) -> None:
        self.calls: list[int] = []

        def record(value: int) -> dict[str, object]:
            self.calls.append(value)
            return {"ok": True, "value": value}

        self.registry = ToolRegistry(
            {
                "record": {
                    "name": "record",
                    "description": "Record one integer value.",
                    "args_model": RecordingArgs,
                    "function": record,
                }
            }
        )

    def test_valid_json_arguments_execute_handler(self) -> None:
        result = self.registry.invoke("record", '{"value": 7}')

        self.assertEqual(result, {"ok": True, "value": 7})
        self.assertEqual(self.calls, [7])

    def test_valid_dictionary_arguments_execute_handler(self) -> None:
        result = self.registry.invoke("record", {"value": 8})

        self.assertEqual(result, {"ok": True, "value": 8})
        self.assertEqual(self.calls, [8])

    def test_unknown_tool_is_rejected(self) -> None:
        with self.assertRaises(UnknownToolError):
            self.registry.invoke("missing", '{"value": 1}')

        self.assertEqual(self.calls, [])

    def test_invalid_json_is_rejected_before_handler_runs(self) -> None:
        with self.assertRaises(InvalidToolArgumentsError):
            self.registry.invoke("record", '{"value":')

        self.assertEqual(self.calls, [])

    def test_missing_required_field_is_rejected_before_handler_runs(self) -> None:
        with self.assertRaises(InvalidToolArgumentsError):
            self.registry.invoke("record", "{}")

        self.assertEqual(self.calls, [])

    def test_wrong_argument_type_is_rejected_before_handler_runs(self) -> None:
        with self.assertRaises(InvalidToolArgumentsError):
            self.registry.invoke("record", '{"value": [1, 2]}')

        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
