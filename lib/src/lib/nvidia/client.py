from __future__ import annotations
import time
from pathlib import Path
from typing import Any,Iterator
from openai import OpenAI
from .models import Model,ModelCatalog,model_id,sanitize_secrets

class NVIDIA:
    """NVIDIA NIM client configured from nvidia-cli models.json."""
    def __init__(self,models_file:str|Path="models.json"):
        self.models_file=Path(models_file); self.catalog=ModelCatalog(self.models_file)
    def reload(self)->None:self.catalog.reload()
    def models(self,*,chat_only:bool=True,safe:bool=True)->list[Model]:
        items=self.catalog.list_models(chat_only)
        return [sanitize_secrets(m) for m in items] if safe else items
    def model(self,identifier:str|None=None,*,safe:bool=True)->Model:
        m=self.catalog.resolve(identifier,chat_only=True)
        return sanitize_secrets(m) if safe else m
    def default_model(self)->str:return model_id(self.catalog.default())
    def resolve_model(self,identifier:str|None=None)->str:return model_id(self.catalog.resolve(identifier))
    def set_default_model(self,identifier:str)->str:
        """Set and persist the catalog default model."""
        return model_id(self.catalog.set_default(identifier))

    def can_run(self,identifier:str|None=None)->bool:
        try:self.catalog.endpoint(self.catalog.resolve(identifier));return True
        except Exception:return False
    def _client_for(self,identifier:str|None=None):
        m=self.catalog.resolve(identifier); base,key=self.catalog.endpoint(m)
        return m,OpenAI(base_url=base,api_key=key)
    @staticmethod
    def _messages(prompt:str,system_prompt:str|None=None,messages:list[dict[str,Any]]|None=None):
        if messages is not None:return messages
        result=[]
        if system_prompt:result.append({"role":"system","content":system_prompt})
        result.append({"role":"user","content":prompt});return result
    def query(self,prompt:str,*,model:str|None=None,system_prompt:str|None=None,messages:list[dict[str,Any]]|None=None,temperature:float=0.2,top_p:float=0.7,max_tokens:int=1024,**kwargs:Any)->str:
        info,client=self._client_for(model)
        c=client.chat.completions.create(model=model_id(info),messages=self._messages(prompt,system_prompt,messages),temperature=temperature,top_p=top_p,max_tokens=max_tokens,stream=False,**kwargs)
        return c.choices[0].message.content or ""
    def stream(self,prompt:str,*,model:str|None=None,system_prompt:str|None=None,messages:list[dict[str,Any]]|None=None,temperature:float=0.2,top_p:float=0.7,max_tokens:int=1024,**kwargs:Any)->Iterator[str]:
        for e in self.stream_events(prompt,model=model,system_prompt=system_prompt,messages=messages,temperature=temperature,top_p=top_p,max_tokens=max_tokens,include_raw=False,**kwargs):
            if e["content"]:yield e["content"]
    def stream_events(self,prompt:str,*,model:str|None=None,system_prompt:str|None=None,messages:list[dict[str,Any]]|None=None,temperature:float=0.2,top_p:float=0.7,max_tokens:int=1024,include_raw:bool=False,**kwargs:Any)->Iterator[dict[str,Any]]:
        info,client=self._client_for(model); started=time.perf_counter(); previous=started
        completion=client.chat.completions.create(model=model_id(info),messages=self._messages(prompt,system_prompt,messages),temperature=temperature,top_p=top_p,max_tokens=max_tokens,stream=True,**kwargs)
        for number,chunk in enumerate(completion,start=1):
            now=time.perf_counter(); content=""; finish=None
            if getattr(chunk,"choices",None):
                choice=chunk.choices[0]; delta=getattr(choice,"delta",None)
                if delta is not None:content=getattr(delta,"content",None) or ""
                finish=getattr(choice,"finish_reason",None)
            event={"chunk":number,"content":content,"chars":len(content),"elapsed_ms":round((now-started)*1000,1),"gap_ms":round((now-previous)*1000,1),"finish_reason":finish}
            if include_raw:
                dump=getattr(chunk,"model_dump",None); event["raw"]=dump() if callable(dump) else None
            previous=now; yield event
