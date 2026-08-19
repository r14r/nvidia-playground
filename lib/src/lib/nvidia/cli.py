import argparse,json
from .client import NVIDIA
from .exceptions import NVIDIAError

def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--file",default="models.json");p.add_argument("--model",default=None);a=p.parse_args()
    try: info=NVIDIA(a.file).model(a.model,safe=True)
    except NVIDIAError as exc: print(f"Error: {exc}"); return 1
    print(json.dumps(info,indent=2,ensure_ascii=False)); return 0
if __name__=="__main__":raise SystemExit(main())
