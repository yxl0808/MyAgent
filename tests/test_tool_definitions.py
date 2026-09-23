import unittest

from agent.tools.bash import BashTool


class ToolDefinitionTest(unittest.TestCase):
    def test_required_flags_become_top_level_json_schema_array(self):
        """验证工具内部必填标记不会泄漏进属性 JSON Schema。"""
        definition = BashTool().to_definition()
        schema = definition["input_schema"]

        self.assertEqual(schema["required"], ["command"])
        self.assertNotIn("required", schema["properties"]["command"])
        self.assertTrue(BashTool.parameters["command"]["required"])


if __name__ == "__main__":
    unittest.main()
