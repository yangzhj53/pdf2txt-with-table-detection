import redis
from typing import Text, Tuple, List, Optional, Any

class RedisClient:
    def __init__(self, host: Text, port: int) -> None:
        # pool = redis.ConnectionPool(host=host, port=port, decode_responses=True)
        self.r = redis.Redis(host=host, port=port, db=0, decode_responses=True)

    def pop(self, list_name: Text) -> Optional[Text]:
        return self.r.rpop(list_name)

    def push(self, list_name: Text, aid: Text):
        return self.r.lpush(list_name, aid)

    def incr(self, key: Text):
        self.r.incr(key, amount=1)

    def get(self, key: Text) -> Any:
        return self.r.get(key)

    def create_aid_list(self, list_name: Text, aid_list: List[Text]):
        aid_list = list(set(aid_list))
        for aid in aid_list:
            self.push(list_name, aid)
        