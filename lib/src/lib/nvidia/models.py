from __future__ import annotations
import copy, json
from pathlib import Path
from typing import Any
from .exceptions import AmbiguousModelError, ModelCatalogError, ModelCredentialError, ModelNotFoundError

Model=dict[str,Any]
Catalog=dict[str,Any]
SENSITIVE_KEYS={"api_key","apiKey","token","access_token","secret","password"}

def model_id(model:Model)->str:
    return str(model.get("id") or model.get("model") or "").strip()

def is_chat_model(model:Model)->bool:
    caps=model.get("capabilities") or {}
    return bool(caps.get("chat")) or model.get("type")=="chat"

def mask_secret(value:str)->str:
    value=str(value or "")
    if not value: return ""
    if len(value)<=12: return "***"
    return f"{value[:8]}…{value[-4:]}"

def sanitize_secrets(value:Any)->Any:
    if isinstance(value,dict):
        return {k:(mask_secret(str(v)) if k in SENSITIVE_KEYS else sanitize_secrets(v)) for k,v in value.items()}
    if isinstance(value,list): return [sanitize_secrets(v) for v in value]
    return copy.deepcopy(value)

class ModelCatalog:
    def __init__(self,path:str|Path="models.json"):
        self.path=Path(path); self.data:Catalog={}; self.reload()
    def reload(self)->None:
        if not self.path.is_file():
            raise ModelCatalogError(f"Model catalog not found: {self.path}. Run `just models` first.")
        try:
            data=json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError) as exc:
            raise ModelCatalogError(f"Could not read model catalog {self.path}: {exc}") from exc
        if not isinstance(data,dict) or not isinstance(data.get("models"),list):
            raise ModelCatalogError("models.json must be an object containing a 'models' array")
        for i,m in enumerate(data["models"]):
            if not isinstance(m,dict): raise ModelCatalogError(f"models[{i}] must be a JSON object")
            if not model_id(m): raise ModelCatalogError(f"models[{i}] has neither 'id' nor 'model'")
        self.data=data
    @property
    def models(self)->list[Model]: return list(self.data.get("models",[]))
    def list_models(self,chat_only:bool=False)->list[Model]:
        return [m for m in self.models if is_chat_model(m)] if chat_only else self.models
    def default(self,chat_only:bool=True)->Model:
        models=self.list_models(chat_only)
        m=next((x for x in models if x.get("default") is True),None)
        if m:return m
        if models:return models[0]
        raise ModelNotFoundError("No suitable models are available")
    def resolve(self,identifier:str|None=None,*,chat_only:bool=True)->Model:
        if not identifier:return self.default(chat_only)
        req=identifier.strip().lower(); models=self.list_models(chat_only)
        exact=[m for m in models if req in {str(m.get("id") or "").lower(),str(m.get("model") or "").lower()}]
        if len(exact)==1:return exact[0]
        seg=[m for m in models if req in [p for p in model_id(m).lower().split("/") if p]]
        if len(seg)==1:return seg[0]
        if len(seg)>1:raise AmbiguousModelError(self._ambiguous(identifier,seg))
        partial=[m for m in models if req in model_id(m).lower()]
        if len(partial)==1:return partial[0]
        if len(partial)>1:raise AmbiguousModelError(self._ambiguous(identifier,partial))
        raise ModelNotFoundError(f"Model '{identifier}' not found. Available models: "+", ".join(model_id(m) for m in models))
    @staticmethod
    def _ambiguous(identifier:str,matches:list[Model])->str:
        return f"Model '{identifier}' is ambiguous. Matches: "+", ".join(model_id(m) for m in matches)
    def save(self)->None:
        """Persist the current catalog atomically."""
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_name(f".{self.path.name}.tmp")
        try:
            temp.write_text(
                json.dumps(self.data,indent=2,ensure_ascii=False)+"\n",
                encoding="utf-8",
            )
            temp.replace(self.path)
        finally:
            if temp.exists():
                temp.unlink()

    def set_default(self,identifier:str)->Model:
        """Set exactly one model as default and persist models.json."""
        selected=self.resolve(identifier,chat_only=False)
        selected_id=model_id(selected)
        for model in self.models:
            model["default"]=model_id(model)==selected_id
        self.save()
        return selected

    @staticmethod
    def endpoint(model:Model)->tuple[str,str]:
        base=str(model.get("base_url") or "").strip(); key=str(model.get("api_key") or "").strip()
        if not base: raise ModelCredentialError(f"Model '{model_id(model)}' has no base_url in models.json")
        if not key: raise ModelCredentialError(f"Model '{model_id(model)}' has no api_key in models.json. Regenerate with --with-api-key.")
        return base,key
    def safe_catalog(self)->Catalog:return sanitize_secrets(self.data)
