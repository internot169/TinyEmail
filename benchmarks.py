import time
import json
import statistics
import logging
import matplotlib.pyplot as plt
from llama import LlamaModel
from deepseek import DeepSeekModel
from email_dataset import load_email_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def benchmark_model(model, dataset):
    start_time = time.time()
    results = []
    for email in dataset:
        try:
            results.append(model.process(email))
        except Exception as e:
            logging.error(f"Error processing email: {e}")
    end_time = time.time()
    return end_time - start_time, results

def evaluate_results(results):
    metrics = {
        "accuracy": statistics.mean([result.get("accuracy", 0) for result in results]),
        "precision": statistics.mean([result.get("precision", 0) for result in results]),
        "recall": statistics.mean([result.get("recall", 0) for result in results]),
        "f1_score": statistics.mean([
            2 * (result.get("precision", 0) * result.get("recall", 0)) /
            (result.get("precision", 0) + result.get("recall", 0) + 1e-9)
            for result in results
        ]),
    }
    return metrics

def save_results_to_file(results, filename):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

def plot_metrics(metrics, model_name):
    labels = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color="skyblue")
    plt.title(f"{model_name} Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.savefig(f"{model_name}_metrics.png")
    plt.close()

def compare_models(llama_metrics, deepseek_metrics):
    comparison = {
        "accuracy_diff": llama_metrics["accuracy"] - deepseek_metrics["accuracy"],
        "precision_diff": llama_metrics["precision"] - deepseek_metrics["precision"],
        "recall_diff": llama_metrics["recall"] - deepseek_metrics["recall"],
        "f1_score_diff": llama_metrics["f1_score"] - deepseek_metrics["f1_score"],
    }
    return comparison

def log_comparison(comparison):
    logging.info("\nModel Comparison:")
    for metric, diff in comparison.items():
        logging.info(f"{metric}: {diff:.2f}")

def main():
    dataset = load_email_dataset("datasets/email_dataset.json")
    logging.info("Email dataset loaded successfully.")

    llama = LlamaModel()
    deepseek = DeepSeekModel()
    logging.info("Models initialized successfully.")

    llama_time, llama_results = benchmark_model(llama, dataset)
    logging.info(f"LLAMA processing time: {llama_time:.2f} seconds")
    llama_metrics = evaluate_results(llama_results)
    logging.info(f"LLAMA metrics: {llama_metrics}")
    save_results_to_file(llama_metrics, "llama_results.json")
    plot_metrics(llama_metrics, "LLAMA")

    deepseek_time, deepseek_results = benchmark_model(deepseek, dataset)
    logging.info(f"DEEPSEEK processing time: {deepseek_time:.2f} seconds")
    deepseek_metrics = evaluate_results(deepseek_results)
    logging.info(f"DEEPSEEK metrics: {deepseek_metrics}")
    save_results_to_file(deepseek_metrics, "deepseek_results.json")
    plot_metrics(deepseek_metrics, "DEEPSEEK")

    comparison = compare_models(llama_metrics, deepseek_metrics)
    log_comparison(comparison)

    print("\nSummary:")
    print(f"LLAMA vs DEEPSEEK processing time: {llama_time:.2f}s vs {deepseek_time:.2f}s")
    print(f"LLAMA accuracy: {llama_metrics['accuracy']:.2f}, DEEPSEEK accuracy: {deepseek_metrics['accuracy']:.2f}")
    print(f"LLAMA precision: {llama_metrics['precision']:.2f}, DEEPSEEK precision: {deepseek_metrics['precision']:.2f}")
    print(f"LLAMA recall: {llama_metrics['recall']:.2f}, DEEPSEEK recall: {deepseek_metrics['recall']:.2f}")
    print(f"LLAMA F1-score: {llama_metrics['f1_score']:.2f}, DEEPSEEK F1-score: {deepseek_metrics['f1_score']:.2f}")

if __name__ == "__main__":
    main()
