import json,tempfile,unittest
from pathlib import Path
from unittest.mock import MagicMock,patch
from lib.nvidia import NVIDIA
SAMPLE={"models":[{"default":True,"id":"minimaxai/minimax-m3","model":"minimaxai/minimax-m3","base_url":"https://example.invalid/v1","api_key":"nvapi-EXAMPLE_MINIMAX_SECRET","type":"chat","capabilities":{"chat":True,"streaming":True}},{"default":False,"id":"meta/llama-3.3-70b-instruct","model":"meta/llama-3.3-70b-instruct","base_url":"https://example.invalid/v1","api_key":"nvapi-EXAMPLE_LLAMA_SECRET","type":"chat","capabilities":{"chat":True,"streaming":True}}]}
class TestNVIDIA(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/"models.json";self.path.write_text(json.dumps(SAMPLE));self.n=NVIDIA(self.path)
    def tearDown(self):self.tmp.cleanup()
    def test_default(self):self.assertEqual(self.n.default_model(),"minimaxai/minimax-m3")
    def test_partial(self):self.assertEqual(self.n.resolve_model("minimax"),"minimaxai/minimax-m3")
    def test_safe(self):self.assertNotEqual(self.n.model("minimax")["api_key"],"nvapi-EXAMPLE_MINIMAX_SECRET")
    def test_set_default_model(self):
        self.assertEqual(
            self.n.set_default_model("llama"),
            "meta/llama-3.3-70b-instruct",
        )
        saved=json.loads(self.path.read_text(encoding="utf-8"))
        defaults=[m["id"] for m in saved["models"] if m.get("default") is True]
        self.assertEqual(defaults,["meta/llama-3.3-70b-instruct"])
        self.assertEqual(
            NVIDIA(self.path).default_model(),
            "meta/llama-3.3-70b-instruct",
        )
    @patch("lib.nvidia.client.OpenAI")
    def test_query(self,OpenAI):
        resp=MagicMock();resp.choices=[MagicMock()];resp.choices[0].message.content="Because molecules scatter blue light.";OpenAI.return_value.chat.completions.create.return_value=resp
        self.assertEqual(self.n.query("why is the sky blue",model="minimax"),"Because molecules scatter blue light.")
        self.assertEqual(OpenAI.return_value.chat.completions.create.call_args.kwargs["model"],"minimaxai/minimax-m3")
if __name__=="__main__":unittest.main()
