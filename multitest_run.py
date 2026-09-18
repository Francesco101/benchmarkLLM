import run_benchmarks as bench

# PARAMETRI CONDIVISI
LIMIT = 50

#-----------------------

all_models = [
    "Qwen/Qwen3.6-35B-A3B-FP8",
    "Qwen/Qwen3.8-27B-FP8",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "google/gemma-4-26B-A4B-it",
    ]

LOGITS_TASKS = [
    "mmlu",
    "arc_challenge",
    "hella_swag",
    "winogrande",
    "mmlu_pro",
    "gpqa_main",
    "gpqa_diamond",
    "musr"
]

CHAT_TASKS = [
    "gsm8k",
    "humaneval",
    "mbpp",
    "ifeval",
    "math500",
    "aime24",
    "livecodebench"
]


# SISTEMO I PARAMETRI (spero vivamente funzioni tutto)

params = bench.benchmarks_parameters

# Settiamo i test logits
params["logits-tests"]["model_args"]["num_concurrent"] = 4      # Perchè di norma sono test più leggeri
params["logits-tests"]["apply_chat_template"] = False
params["logits-tests"]["gen_kwargs"] = {}    # Non serve nulla 
#params["logits-tests"]["tasks"] = LOGITS_TASKS  # RIMUOVI DOPO DEBUG, per ora meglio fare uno alla volta

# Settiamo i test a risposta completa
params["chat-tests"]["model_args"]["num_concurrent"] = 1    # Forse va bene 2 ? ma restiamo conservativi
params["chat-tests"]["apply_chat_template"] = True
params["chat-tests"]["gen_kwargs"]["max_tokens"] = 8192
#params["chat-tests"]["tasks"] = CHAT_TASKS

# Parametri condivisi
bench.set_limit(LIMIT)


for model in all_models:

    # Indica il modello in in questione
    bench.set_llm(model)

    # Definiamo gli output dir e file
    output_path = "./benchmark_outputs/multitest/" + model + "/"
    bench.os.makedirs(output_path, exist_ok=True)
    # inizializzo il file che tiene traccia dei fallimenti
    errorfile = output_path + "failed-tasks"
    with open(errorfile, 'w') as file:
        file.write('task, model, task_type')

    for i, TASKS in enumerate([LOGITS_TASKS, CHAT_TASKS]):
        task_type = ["logits-tests", "chat-tests"][i]
        for task in TASKS:
            params[task_type]["tasks"] = [task]
            # Eseguo il benchmark, se qualcosa va storto lo riporto
            try: 
                output = bench.run_benchmark(params[task_type])
            except:
                output = None
                file = open(errorfile, 'a')
                file.write("%s,%s,%s\n" % (task, model, task_type))
                file.close()

            # Creiamo il file con i risultati
            results_file = task + "__" + task_type + ".json"

            # Se tutto è andato bene esporto i risultati
            if output is not None:
                bench.output_results(output, output_path=output_path, results_file=results_file, silent=True)
