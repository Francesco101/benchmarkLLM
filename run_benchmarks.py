import lm_eval
import json
import os
import datetime


def __main__():

    tested_model = "Qwen/Qwen3.6-35B-A3B-FP8"
    set_llm(tested_model)
    set_limit(5)

    # aggiorno a solo dei subset tasks di mmlu perché mmlu è troppo lungo:
#    benchmarks_parameters["mmlu"]["tasks"] =  ["mmlu_anatomy", 
#         "mmlu_anatomy_generative_spanish", 
#         "mmlu_abstract_algebra"
#        ]
 
    benchmarks_parameters["gsm8k"]["tasks"] =  ["hendrycks_math",
                                               "minerva_math"]

    res = run_benchmark(benchmarks_parameters["gsm8k"])
    output_results(res)

    return 0

#-----------------------------------------
def disable_tokenizers_parallelism():
    # Disabilita il parallelismo dei tokenizer per evitare il warning di fork
    # Nota: viene causato da mmlu, ma non crea problemi.
    # RIMANE DA CAPIRE COMUNQUE
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
#------------------------------------------


# Qui sotto metto alcune funzioni che agiscono su tutte le impostazioni per i benchmark
# facevo prima a fare delle classi probabilmente ma vabbé per ora è così.
def set_limit(limit: int):
    for b in benchmarks_parameters.keys():
         benchmarks_parameters[b]["limit"] = limit


def set_llm(model: str, tokenizer=None):
    for b in benchmarks_parameters.keys():
         benchmarks_parameters[b]["model_args"]["model"] = model
         if tokenizer is not None:
            benchmarks_parameters[b]["model_args"]["tokenizer"] = tokenizer

#-------------------------------------------------------------------------------

def run_benchmark(parameters: dict):

    if parameters["model_args"]["model"] is None:
         raise InterruptedError("LLM-model not specified")

    # bemchmark
    results = lm_eval.simple_evaluate(
        model=parameters["model"],
        model_args=parameters["model_args"],
        gen_kwargs=parameters["gen_kwargs"],
        apply_chat_template=parameters["apply_chat_template"],
        tasks=parameters["tasks"],
        limit=parameters["limit"],
        log_samples=parameters["log_samples"],
    )
    return results


def output_results(results, output_path="./benchmark_outputs", results_file=None, samples_file=None):
    # creazione cartella per gli output
    os.makedirs(output_path, exist_ok=True)

    t = datetime.datetime.now()
    date = "%d-%d-%dT%d-%d-%d" % (t.year, t.month, t.day, t.hour, t.minute, t.second) 
    if results_file is None:
        results_file = date + "_results.json"
    if samples_file is None:
        samples_file = date + "_samples.json"

    # Salvataggio dei file JSON (equivalente a --output_path e --log_samples)
    with open(os.path.join(output_path, results_file), "w", encoding="utf-8") as f:
        json.dump(results["results"], f, indent=2, ensure_ascii=False)

    if "samples" in results and results["samples"]:
        with open(os.path.join(output_path, samples_file), "w", encoding="utf-8") as f:
            json.dump(results["samples"], f, indent=2, ensure_ascii=False)

    print(f"Benchmark completato con successo. Risultati salvati in '{output_path}'.")



# DEFINISCO I PARAMETRI PER I BENCHMARK
# TODO: poi sarà da tenerli su un json esterno e tenerne bene traccia
#       \_ per ora sto cercando di capire come funziona.
#       \_ le prime chiavi benchmarks_parameters.keys() non hanno nessun valore
#       \_ saranno da cambiare, ma per ora significano: "questo setting funziona per X eval task"

benchmarks_parameters = {
    "mmlu": {
            "model": "local-completions",
            "model_args": {
                "model": None,
                "base_url": "https://orfeo-llm.areasciencepark.it/vllm/v1/completions",
                "num_concurrent": 1
            },
            "gen_kwargs": {
                "max_tokens": 8192,
                "temperature": 0.0,
                "until": []
            },
            "apply_chat_template": True,
            "tasks": ["mmlu"],
            "limit": 10,   # None per eseguire il benchmark completo
            "log_samples": False

    },

    "gsm8k": {
        "model": "local-chat-completions",
        "model_args": {
            "model": None,
            "tokenizer": False,
            "base_url": "https://orfeo-llm.areasciencepark.it/vllm/v1/chat/completions",
            "num_concurrent": 1
        },
        "gen_kwargs": {
            "max_tokens": 8192,
            "temperature": 0.0,
            "until": []
        },
        "apply_chat_template": True,
        "tasks": ["gsm8k"],
        "limit": 10,   # None per eseguire il benchmark completo
        "log_samples": False

    }
}

__main__()