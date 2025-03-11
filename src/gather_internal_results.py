import redis
import pandas
import os
from tqdm import tqdm

r = redis.Redis(host=os.environ["REDIS_SERVER"], port=os.environ["REDIS_PORT"], db=0)


results = []
while True:
    for key in tqdm(r.scan_iter("*real_toxic_prompts*", count=10000000)):

        path = key.decode('UTF-8')


        elements = path.split("/")

        project = elements[1]
        model_name = elements[2]
        chat_template_model_name = elements[3]
        encoding = elements[4]
        control_task_type = elements[5]
        probe_type = elements[6]
        generation_id = int(elements[7].split(":")[1])
        template_index = int(elements[8].split(":")[1])
        sample_size = int(elements[9].split(":")[1])
        origin = elements[10]
        layer_id = int(elements[11].split(":")[1])
        seed = int(elements[12].split(":")[1])
        fold = int(elements[13].split(":")[1][0])

        if "cross" not in project and "back" not in project:
            task = project.split("-")[-1]
        elif "cross" in project:
            task = "forward-" + project.split("-")[-1]
        else:
            task = "backward-" + project.split("-")[-1]

        result = r.hgetall(key)
        result = {
            k.decode('UTF-8'):v.decode('UTF-8')
            for k, v in result.items()
        }

        entry = {
            "task": task,
            "model_name": model_name,
            "chat_template_model_name": chat_template_model_name,
            "seed": seed,
            "fold": fold,
            "probe_type": probe_type,
            "layer_id": layer_id,
            "template_index": template_index,
            "control_task_type": control_task_type,
            "encoding": encoding,
            "generation_id": generation_id,
            "origin": origin,
        }
        if "compression" in result:
            entry["compression"] = float(result["compression"])
        if "score" in result:
            entry["score"] = float(result["score"])
        if "score_acc" in result:
            entry["score_acc"] = float(result["score_acc"])
        if "score acc" in result:
            entry["score_acc"] = float(result["score acc"])
        if "full_test_pearson" in result:
            entry["score"] = float(result["full_test_pearson"])
        if "full_test_f1" in result:
            entry["score"] = float(result["full_test_f1"])
        if "full_test_acc" in result:
            entry["score_acc"] = float(result["full_test_acc"])
        if "full_test_error" in result:
            entry["error"] = float(result["full_test_error"])

        results.append(entry)

    (pandas.DataFrame(results)
    .drop_duplicates([
        "task", "model_name", "chat_template_model_name",
        "seed", "fold", "layer_id", "template_index", "control_task_type",
        "encoding", "generation_id", "origin"
    ])
     .to_csv("/nas/lift/internal_scores.csv"))