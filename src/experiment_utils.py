import random
import redis

def get_encoding_path(
        encoding_folder, task, model_name, chat_template_model_name, model_precision,
        template_index, num_return_sequences, skip_layers=None
):
    if skip_layers:
        model_name = f"{model_name}-wo-{skip_layers}"

    return f"{encoding_folder}/{task}/{model_name.replace('/', '_')}/{chat_template_model_name.replace('/', '_')}/{model_precision}/template-{template_index}/num-gens{num_return_sequences}/"


def check_redis_path(fields, host=os.environ["REDIS_SERVER"], ports=[os.environ["REDIS_PORT"]]):
    lookup_port = random.choice(ports)

    path = "/" +  "/".join([
        str(v)
        for k, v in fields.items()
    ]) + "/*"

    r = redis.Redis(host=host, port=lookup_port, db=0)

    print("check for", path)
    match_keys = r.scan(cursor=0, match=path, count=100000000)[1]

    return len(match_keys)