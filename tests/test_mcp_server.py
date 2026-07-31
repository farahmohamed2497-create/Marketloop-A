import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_server.server import MarketLoopMCPServer


class MarketLoopMCPServerTests(unittest.TestCase):
    def test_initialize_declares_capabilities(self) -> None:
        server = MarketLoopMCPServer()
        payload = server.initialize()
        self.assertEqual(payload["serverInfo"]["name"], "marketloop")
        self.assertEqual(payload["capabilities"]["tools"], {"listChanged": False})
        self.assertEqual(payload["capabilities"]["resources"], {"subscribe": False, "listChanged": False})
        self.assertEqual(payload["capabilities"]["prompts"], {"listChanged": False})
        self.assertEqual(payload["capabilities"]["logging"], {})

    def test_connect_registers_discovered_modules(self) -> None:
        server = MarketLoopMCPServer()
        info = server.connect()
        self.assertEqual(info["transport"], "stdio")
        self.assertTrue(info["database"].endswith("marketloop.db"))
        self.assertTrue({"inventory_tool", "order_tool", "update_inventory_quantity", "process_return_request"}.issubset(set(server.list_tools())))
        self.assertEqual(
            server.list_resources(),
            ["catalog_resource", "return_and_refund_resource", "shipping_sla_resource"],
        )
        self.assertEqual(server.list_prompts(), ["draft_return_response", "product_prompt"])

    def test_list_tools_includes_input_schema(self) -> None:
        server = MarketLoopMCPServer()
        server.register_modules()
        result = asyncio.run(server._list_tools(None))
        self.assertTrue(result.tools)
        schema_values = [getattr(tool, "inputSchema", getattr(tool, "input_schema", None)) for tool in result.tools]
        self.assertTrue(all(schema is not None for schema in schema_values))

    def test_list_resources_exposes_policy_uris(self) -> None:
        server = MarketLoopMCPServer()
        server.register_modules()
        result = asyncio.run(server._list_resources(None))
        uris = {resource.uri for resource in result.resources}
        self.assertIn("marketloop://policies/return_and_refund", uris)
        self.assertIn("marketloop://policies/shipping_sla", uris)
        self.assertEqual({resource.mime_type for resource in result.resources}, {"text/markdown"})

    def test_read_return_and_refund_resource(self) -> None:
        server = MarketLoopMCPServer()
        server.register_modules()
        params = type("Params", (), {"uri": "marketloop://policies/return_and_refund"})()
        result = asyncio.run(server._read_resource(None, params))
        text = result.contents[0].text
        self.assertIn("30 calendar days", text)
        self.assertIn("15% restocking fee", text)
        self.assertIn("5-10 business days", text)

    def test_read_shipping_sla_resource(self) -> None:
        server = MarketLoopMCPServer()
        server.register_modules()
        params = type("Params", (), {"uri": "marketloop://policies/shipping_sla"})()
        result = asyncio.run(server._read_resource(None, params))
        text = result.contents[0].text
        self.assertIn("3-5 business days", text)
        self.assertIn("7 calendar days", text)

    def test_read_unknown_resource_raises(self) -> None:
        server = MarketLoopMCPServer()
        server.register_modules()
        params = type("Params", (), {"uri": "marketloop://policies/nonexistent"})()
        with self.assertRaises(ValueError):
            asyncio.run(server._read_resource(None, params))

    def test_list_prompts_declares_draft_return_response_arguments(self) -> None:
        server = MarketLoopMCPServer()
        server.register_modules()
        result = asyncio.run(server._list_prompts(None))
        by_name = {prompt.name: prompt for prompt in result.prompts}
        self.assertIn("draft_return_response", by_name)
        arg_names = {argument.name for argument in by_name["draft_return_response"].arguments}
        self.assertEqual(arg_names, {"order_id", "customer_name", "decision_status", "reason"})

    def test_get_prompt_formats_approved_email(self) -> None:
        server = MarketLoopMCPServer()
        server.register_modules()
        params = type(
            "Params",
            (),
            {
                "name": "draft_return_response",
                "arguments": {
                    "order_id": 42,
                    "customer_name": "Ali Mahmoud",
                    "decision_status": "Approved",
                    "reason": "Item arrived damaged",
                },
            },
        )()
        result = asyncio.run(server._get_prompt(None, params))
        text = result.messages[0].content.text
        self.assertIn("Order #42", text)
        self.assertIn("Ali Mahmoud", text)
        self.assertIn("**Approved**", text)
        self.assertIn("prepaid return label", text.lower())

    def test_get_prompt_formats_rejected_email(self) -> None:
        server = MarketLoopMCPServer()
        server.register_modules()
        params = type(
            "Params",
            (),
            {
                "name": "draft_return_response",
                "arguments": {
                    "order_id": 7,
                    "customer_name": "Mariam Ahmed",
                    "decision_status": "Rejected",
                    "reason": "The 30-day return window has passed",
                },
            },
        )()
        result = asyncio.run(server._get_prompt(None, params))
        text = result.messages[0].content.text
        self.assertIn("Order #7", text)
        self.assertIn("Mariam Ahmed", text)
        self.assertIn("**Rejected**", text)
        self.assertIn("The 30-day return window has passed", text)

    def test_get_prompt_rejects_invalid_decision_status(self) -> None:
        server = MarketLoopMCPServer()
        server.register_modules()
        params = type(
            "Params",
            (),
            {
                "name": "draft_return_response",
                "arguments": {
                    "order_id": 1,
                    "customer_name": "Ali Mahmoud",
                    "decision_status": "Maybe",
                    "reason": "test",
                },
            },
        )()
        with self.assertRaises(ValueError):
            asyncio.run(server._get_prompt(None, params))

    def test_get_unknown_prompt_raises(self) -> None:
        server = MarketLoopMCPServer()
        server.register_modules()
        params = type("Params", (), {"name": "missing_prompt", "arguments": {}})()
        with self.assertRaises(ValueError):
            asyncio.run(server._get_prompt(None, params))


if __name__ == "__main__":
    unittest.main()
