#!/usr/bin/env python3
"""
Main script for running Tau2Bench evaluations with unified model support.

This script provides a command-line interface for evaluating models on Tau2Bench,
supporting both VLLM-hosted models and API models (OpenAI, etc.).

Example usage:
    # Evaluate VLLM-hosted model
    python run_tau2_eval.py \\
        --model-name my_model \\
        --backend vllm \\
        --base-url http://localhost:8500/v1 \\
        --experiment-name my_model_test

    # Evaluate GPT-4
    python run_tau2_eval.py \\
        --model-name gpt-4o \\
        --backend openai \\
        --experiment-name gpt4o_test

    # Analyze results
    python run_tau2_eval.py --analyze --experiment my_model_test
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from evaluator import Tau2BenchEvaluator, Tau2EvalConfig
from analyze import Tau2BenchAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Tau2Bench evaluation with unified model support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--evaluate", action="store_true",
                           help="Run evaluation mode")
    mode_group.add_argument("--analyze", action="store_true",
                           help="Run analysis mode")

    # Model configuration (for evaluation)
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model-name", type=str,
                            help="Name or identifier of the model")
    model_group.add_argument("--backend", type=str, choices=["vllm", "openai", "litellm", "auto"],
                            default="auto", help="Model backend")
    model_group.add_argument("--base-url", type=str,
                            help="Base URL for model API (for VLLM or custom endpoints)")
    model_group.add_argument("--api-key", type=str,
                            help="API key (if required)")
    model_group.add_argument("--temperature", type=float, default=0.0,
                            help="Sampling temperature for agent model")
    model_group.add_argument("--max-tokens", type=int, default=8192,
                            help="Maximum tokens for generation")

    # Evaluation configuration
    eval_group = parser.add_argument_group("Evaluation Configuration")
    eval_group.add_argument("--domains", nargs="+", default=["retail", "airline", "telecom"],
                           help="Domains to evaluate (default: all)")
    eval_group.add_argument("--task-split", type=str, default="test",
                           choices=["train", "test", "base"],
                           help="Task split to use (IMPORTANT: use 'test' for evaluation)")
    eval_group.add_argument("--num-trials", type=int, default=4,
                           help="Number of trials per task (for Pass@k)")
    eval_group.add_argument("--max-steps", type=int, default=100,
                           help="Maximum conversation turns")
    eval_group.add_argument("--max-concurrency", type=int, default=8,
                           help="Maximum concurrent evaluations")

    # User simulator configuration
    user_group = parser.add_argument_group("User Simulator Configuration")
    user_group.add_argument("--user-model", type=str, default="gpt-4o-mini",
                           help="Model for user simulator")
    user_group.add_argument("--user-backend", type=str, default="openai",
                           choices=["openai", "litellm"],
                           help="Backend for user simulator")
    user_group.add_argument("--user-base-url", type=str,
                           help="Base URL for user simulator (e.g., )")
    user_group.add_argument("--user-api-key", type=str,
                           help="API key for user simulator")
    user_group.add_argument("--user-temperature", type=float, default=1.0,
                           help="Temperature for user simulator")

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument("--output-dir", type=str, default="outputs/tau2bench",
                             help="Directory for output files")
    output_group.add_argument("--experiment-name", type=str,
                             help="Name for this experiment (auto-generated if not provided)")

    # Analysis configuration
    analysis_group = parser.add_argument_group("Analysis Configuration")
    analysis_group.add_argument("--experiments", nargs="+",
                               help="Experiment names to analyze (default: all)")
    analysis_group.add_argument("--generate-plots", action="store_true",
                               help="Generate plots during analysis")
    analysis_group.add_argument("--generate-tables", action="store_true",
                               help="Generate tables during analysis")
    analysis_group.add_argument("--export-excel", action="store_true",
                               help="Export results to Excel")

    # Tau2 configuration
    tau2_group = parser.add_argument_group("Tau2Bench Configuration")
    tau2_group.add_argument("--tau2-data-dir", type=str,
                           help="Path to tau2-bench/data directory")
    tau2_group.add_argument("--seed", type=int, default=42,
                           help="Random seed")

    return parser.parse_args()


def run_evaluation(args):
    """Run Tau2Bench evaluation."""
    print("\n" + "="*80)
    print("TAU2BENCH EVALUATION")
    print("="*80 + "\n")

    # Validate required arguments
    if not args.model_name:
        print("Error: --model-name is required for evaluation mode")
        sys.exit(1)

    # Create configuration
    config = Tau2EvalConfig(
        model_name=args.model_name,
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        domains=args.domains,
        task_split=args.task_split,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_concurrency=args.max_concurrency,
        user_model_name=args.user_model,
        user_backend=args.user_backend,
        user_base_url=args.user_base_url,
        user_api_key=args.user_api_key,
        user_temperature=args.user_temperature,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        tau2_data_dir=args.tau2_data_dir,
        seed=args.seed,
    )

    # Create evaluator
    evaluator = Tau2BenchEvaluator(config)

    # Run evaluation
    try:
        results = evaluator.run_evaluation()

        # Compute metrics
        if results["trajectory_files"]:
            print("\nComputing metrics...")
            metrics = evaluator.compute_metrics(results["trajectory_files"])

            # Display summary
            print("\n" + "="*80)
            print("EVALUATION SUMMARY")
            print("="*80)
            for domain, domain_metrics in metrics.items():
                print(f"\n{domain.upper()} Domain:")
                for k in [1, 2, 3, 4]:
                    if f"pass@{k}" in domain_metrics:
                        print(f"  Pass@{k}: {domain_metrics[f'pass@{k}']*100:.1f}%")
                print(f"  Avg Reward: {domain_metrics.get('average_reward', 0):.3f}")
                print(f"  Num Tasks: {domain_metrics.get('num_tasks', 0)}")

        print("\n✓ Evaluation complete!")
        print(f"✓ Results saved to: {evaluator.output_dir}")

    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_analysis(args):
    """Run analysis on evaluation results."""
    print("\n" + "="*80)
    print("TAU2BENCH ANALYSIS")
    print("="*80 + "\n")

    # Create analyzer
    analyzer = Tau2BenchAnalyzer(results_dir=args.output_dir)

    # Load model results
    analyzer.load_all_models(experiment_names=args.experiments)

    if not analyzer.models_data:
        print("No model results found!")
        sys.exit(1)

    print(f"\nLoaded {len(analyzer.models_data)} model(s)")

    # Generate summary table
    print("\n" + "="*80)
    print("PASS@4 SUMMARY")
    print("="*80)
    summary_table = analyzer.get_summary_table(metric="pass@4", format_type="markdown")
    print(summary_table)

    # Generate plots if requested
    if args.generate_plots:
        print("\nGenerating plots...")
        plots_dir = Path(args.output_dir) / "analysis_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Pass@k comparison
        analyzer.plot_passk_comparison(
            save_path=str(plots_dir / "passk_comparison.png")
        )

        # Domain comparison
        analyzer.plot_domain_comparison(
            metric="pass@4",
            save_path=str(plots_dir / "domain_comparison.png")
        )

        print(f"✓ Plots saved to {plots_dir}")

    # Generate tables if requested
    if args.generate_tables:
        print("\nGenerating tables...")
        tables_dir = Path(args.output_dir) / "analysis_tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        model_names = list(analyzer.models_data.keys())

        # LaTeX table
        analyzer.generate_paper_table(
            model_names=model_names,
            output_file=str(tables_dir / "results_table.tex"),
            format="latex"
        )

        # Markdown table
        analyzer.generate_paper_table(
            model_names=model_names,
            output_file=str(tables_dir / "results_table.md"),
            format="markdown"
        )

        print(f"✓ Tables saved to {tables_dir}")

    # Export to Excel if requested
    if args.export_excel:
        print("\nExporting to Excel...")
        excel_path = Path(args.output_dir) / "tau2bench_results.xlsx"
        analyzer.export_to_excel(str(excel_path))
        print(f"✓ Excel file saved to {excel_path}")

    print("\n✓ Analysis complete!")


def main():
    args = parse_args()

    if args.evaluate:
        run_evaluation(args)
    elif args.analyze:
        run_analysis(args)


if __name__ == "__main__":
    main()
