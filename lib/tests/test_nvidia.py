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
    @patch("lib.nvidia.client.OpenAI")
    def test_query(self,OpenAI):
        resp=MagicMock();resp.choices=[MagicMock()];resp.choices[0].message.content="Because molecules scatter blue light.";OpenAI.return_value.chat.completions.create.return_value=resp
        self.assertEqual(self.n.query("why is the sky blue",model="minimax"),"Because molecules scatter blue light.")
    @patch("lib.nvidia.client.OpenAI")
    def test_query_omits_zero_top_p(self,OpenAI):
        resp=MagicMock();resp.choices=[MagicMock()];resp.choices[0].message.content="ok";OpenAI.return_value.chat.completions.create.return_value=resp
        self.assertEqual(self.n.query("hello",model="minimax",top_p=0),"ok")
        self.assertNotIn("top_p",OpenAI.return_value.chat.completions.create.call_args.kwargs)
    def test_content_text_parts(self):
        self.assertEqual(NVIDIA._content_text([{"type":"text","text":"Hello "},{"type":"text","text":"world"}]),"Hello world")

class TestNVIDIAAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/"models.json";self.path.write_text(json.dumps(SAMPLE));self.n=NVIDIA(self.path)
    def tearDown(self):self.tmp.cleanup()
    async def test_async_query_preserves_full_response(self):
        expected="This is a complete multi word response."
        with patch.object(self.n,"query",return_value=expected):
            result=await self.n.async_query("hello",model="minimax")
        self.assertEqual(result,expected)
    async def test_async_stream_preserves_all_events(self):
        expected=[{"chunk":1,"content":"Hello ","chars":6,"elapsed_ms":1.0,"gap_ms":1.0,"finish_reason":None},{"chunk":2,"content":"from ","chars":5,"elapsed_ms":2.0,"gap_ms":1.0,"finish_reason":None},{"chunk":3,"content":"NVIDIA","chars":6,"elapsed_ms":3.0,"gap_ms":1.0,"finish_reason":"stop"}]
        with patch.object(self.n,"stream_events",return_value=iter(expected)):
            actual=[]
            async for event in self.n.async_stream_events("hello",model="minimax"):
                actual.append(event)
        self.assertEqual(actual,expected)
        self.assertEqual("".join(e["content"] for e in actual),"Hello from NVIDIA")

if __name__=="__main__":unittest.main()
