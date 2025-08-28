import json, pathlib
from typing import List, Dict, Any
from typing import Union

def ensure_dir(p: Union[str, pathlib.Path]):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


class FSBus:
    def __init__(self, topics_dir: str):
        self.topics_dir = pathlib.Path(topics_dir)
        ensure_dir(self.topics_dir)

    def topic_path(self, topic: str) -> pathlib.Path:
        return self.topics_dir / f"{topic}.jsonl"

    def produce(self, topic: str, record: Dict[str, Any]):
        p = self.topic_path(topic)
        ensure_dir(p.parent)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all(self, topic: str) -> List[Dict[str, Any]]:
        p = self.topic_path(topic)
        if not p.exists():
            return []
        out = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line:
                    out.append(json.loads(line))
        return out
