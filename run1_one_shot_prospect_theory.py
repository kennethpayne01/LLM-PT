import json
import os
import time
import random
from datetime import datetime
import anthropic
import openai
import google.generativeai as genai
from dotenv import load_dotenv
import csv

# Load environment variables
load_dotenv()

# Advanced Model Configuration
MODEL_CONFIG = {
    "o3-2025-04-16": {
        "client": "openai",
        "model_name": "o3-2025-04-16",
        "max_tokens": 2000,
        "temperature": None  # o3 only supports default temperature (1.0)
    },
    "claude-sonnet-4": {
        "client": "anthropic", 
        "model_name": "claude-sonnet-4-20250514",
        "max_tokens": 2000,
        "temperature": 0.7
    },
    "claude-sonnet-4-thinking": {
        "client": "anthropic",
        "model_name": "claude-sonnet-4-20250514",
        "max_tokens": 12000,  # Must be > thinking.budget_tokens (10000)
        "temperature": 1,  # Thinking mode requires temperature=1
        "use_thinking": True
    },
    "gpt-4o": {
        "client": "openai",
        "model_name": "gpt-4o",
        "max_tokens": 2000,
        "temperature": 0.7
    },
    "gemini-1.5-pro": {
        "client": "google",
        "model_name": "gemini-1.5-pro",
        "max_tokens": 2000,
        "temperature": 0.7
    }
}

class OneShootProspectTheoryExperiment:
    def __init__(self, models=None, runs_per_condition=2, experiment_version="one_shot_v1", force_fresh_start=False):
        self.models = models or ["o3-2025-04-16", "claude-sonnet-4", "gemini-1.5-pro"]
        self.runs_per_condition = runs_per_condition
        self.experiment_version = experiment_version
        self.force_fresh_start = force_fresh_start
        
        # Initialize API clients
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        # Load scenarios adapted for one-shot decisions
        self.scenarios = self.load_one_shot_scenarios()
        
        # Track results
        self.results = []
        self.intermediate_results = []
        
        # Check for resume capability (unless forcing fresh start)
        if force_fresh_start:
            print("FORCE FRESH START: Ignoring any previous intermediate results")
            self.resume_from_condition = 0
        else:
            self.resume_from_condition = self.check_for_resume()
        
        # Experiment configuration
        self.experiment_config = {
            "version": experiment_version,
            "description": "Pure one-shot prospect theory test with advanced reasoning models",
            "models": self.models,
            "scenarios": list(self.scenarios.keys()),
            "runs_per_condition": runs_per_condition,
            "total_conditions": len(self.models) * 2 * len(self.scenarios) * runs_per_condition,  # models × frames × scenarios × runs (now includes French scenario)
            "timestamp": datetime.now().isoformat()
        }
        
        if self.resume_from_condition > 0:
            print(f"RESUMING experiment from condition {self.resume_from_condition + 1}")
        else:
            print(f"Initialized One-Shot Prospect Theory Experiment")
        print(f"Models: {self.models}")
        print(f"Scenarios: {list(self.scenarios.keys())}")
        print(f"Total conditions: {self.experiment_config['total_conditions']}")

    def check_for_resume(self):
        """
        Check for existing intermediate results and determine where to resume from
        """
        import glob
        
        # Find all intermediate result files
        intermediate_files = glob.glob("run1_intermediate_results_*.json")
        
        if not intermediate_files:
            print("No previous intermediate results found. Starting fresh.")
            return 0
        
        # Sort by condition number to find the latest
        def extract_condition_number(filename):
            try:
                parts = filename.split('_')
                for part in parts:
                    if part.isdigit():
                        return int(part)
                return 0
            except:
                return 0
        
        intermediate_files.sort(key=extract_condition_number)
        latest_file = intermediate_files[-1]
        
        try:
            with open(latest_file, 'r') as f:
                previous_results = json.load(f)
            
            if previous_results:
                # Load previous results
                self.results = previous_results
                last_condition = max(result['condition_number'] for result in previous_results)
                
                print(f"Found previous results in {latest_file}")
                print(f"Last completed condition: {last_condition}")
                print(f"Resuming from condition {last_condition + 1}")
                
                return last_condition
            
        except Exception as e:
            print(f"Error reading {latest_file}: {e}")
            print("Starting fresh.")
            return 0
        
        return 0

    def load_one_shot_scenarios(self):
        """
        Load one-shot prospect theory scenarios from updated JSON files with explicit payoffs
        """
        scenario_files = {
            "business_merger": "business_merger_v3.json",
            "career_transition": "career_transition_v3.json", 
            "championship_strategy": "championship_strategy_v3.json"
        }
        
        scenarios = {}
        for scenario_name, file_path in scenario_files.items():
            try:
                with open(file_path, 'r') as f:
                    scenario_data = json.load(f)
                    scenarios[scenario_name] = scenario_data
            except FileNotFoundError:
                print(f"Warning: Could not find {file_path}. Creating fallback scenario.")
                scenarios[scenario_name] = self.create_fallback_scenario(scenario_name)
        
        return scenarios

    def create_fallback_scenario(self, scenario_name):
        """
        Create a fallback scenario if JSON files aren't found
        """
        # This is a backup in case the files aren't found
        return {
            "context": f"Fallback scenario for {scenario_name}",
            "loss_frame": {
                "overall": "Fallback loss frame",
                "option1": {"description": "Fallback option 1", "payoff": "100% chance of preventing 0.5 points of decline"},
                "option2": {"description": "Fallback option 2", "payoff": "50% chance of preventing 1.0 points of decline, 50% chance of preventing 0.0 points"},
                "option3": {"description": "Fallback option 3", "payoff": "25% chance of preventing 2.0 points of decline, 75% chance of preventing 0.0 points"}
            },
            "gain_frame": {
                "overall": "Fallback gain frame", 
                "option1": {"description": "Fallback option 1", "payoff": "100% chance of gaining 0.5 points"},
                "option2": {"description": "Fallback option 2", "payoff": "50% chance of gaining 1.0 points, 50% chance of gaining 0.0 points"},
                "option3": {"description": "Fallback option 3", "payoff": "25% chance of gaining 2.0 points, 75% chance of gaining 0.0 points"}
            }
        }

    def make_decision(self, model, scenario_name, frame, run_number):
        """
        Get a single decision from a model for a scenario
        """
        scenario = self.scenarios[scenario_name]
        frame_data = scenario[frame]
        
        # Create the one-shot decision prompt 
        # Get the option names dynamically from the scenario structure
        option_names = list(frame_data.keys())
        option_names.remove('overall')  # Remove the overall description
        
        prompt = f"""
{scenario['context']}

**CURRENT SITUATION**:
{frame_data['overall']}

**AVAILABLE OPTIONS**:
"""
        
        # Add options dynamically
        for i, option_name in enumerate(option_names, 1):
            option_data = frame_data[option_name]
            prompt += f"""
{i}. **{option_name.upper()}**:
   - {option_data['description']}
   - {option_data['payoff']}
"""
        
        # Dynamic decision options for the prompt
        option_list = "/".join([opt.upper() for opt in option_names])
        
        prompt += f"""
Please carefully analyze the situation and choose the option you believe is most appropriate.

Provide your reasoning and end your response with:
DECISION: [{option_list}]
        """

        config = MODEL_CONFIG[model]
        
        try:
            if config["client"] == "openai":
                # Check if this is an o-series model that requires max_completion_tokens
                if config["model_name"].startswith("o3") or config["model_name"].startswith("o4"):
                    # o-series models have parameter restrictions: max_completion_tokens and default temperature only
                    response = self.openai_client.chat.completions.create(
                        model=config["model_name"],
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=config["max_tokens"]
                        # Note: o-series models only support default temperature (1.0)
                    )
                else:
                    response = self.openai_client.chat.completions.create(
                        model=config["model_name"],
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=config["max_tokens"],
                        temperature=config["temperature"]
                    )
                response_text = response.choices[0].message.content
                
            elif config["client"] == "anthropic":
                # Handle thinking version of Claude
                if config.get("use_thinking", False):
                    response = self.anthropic_client.messages.create(
                        model=config["model_name"],
                        max_tokens=config["max_tokens"],
                        temperature=config["temperature"],
                        messages=[{"role": "user", "content": prompt}],
                        thinking={
                            "type": "enabled",
                            "budget_tokens": 10000
                        }
                    )
                else:
                    response = self.anthropic_client.messages.create(
                        model=config["model_name"],
                        max_tokens=config["max_tokens"],
                        temperature=config["temperature"],
                        messages=[{"role": "user", "content": prompt}]
                    )
                # Handle response text extraction for both regular and thinking modes
                if config.get("use_thinking", False):
                    # Thinking mode response structure is different
                    response_text = ""
                    for content_block in response.content:
                        if hasattr(content_block, 'text'):
                            response_text += content_block.text
                        elif hasattr(content_block, 'content'):
                            response_text += content_block.content
                else:
                    response_text = response.content[0].text
                
            elif config["client"] == "google":
                model = genai.GenerativeModel(config["model_name"])
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=config["max_tokens"],
                        temperature=config["temperature"]
                    )
                )
                response_text = response.text
            
            # Extract decision
            decision = None
            if "DECISION:" in response_text or "DÉCISION:" in response_text:
                if "DÉCISION:" in response_text:
                    decision_line = response_text.split("DÉCISION:")[-1].strip()
                    # Handle French decisions
                    french_options = {"NÉGOCIER": "negotiate", "ÉGALER": "match", "ESCALADER": "escalate"}
                    for french_option, english_option in french_options.items():
                        if french_option in decision_line.upper():
                            decision = english_option
                            break
                else:
                    decision_line = response_text.split("DECISION:")[-1].strip()
                    # Check for civilian scenario actions
                    civilian_actions = [
                        "ACCEPT", "COUNTER", "DEMAND",  # Business
                        "APPLY", "NETWORK", "STARTUP",  # Career  
                        "CONTROL", "ATTACK", "BLITZ"    # Sports
                    ]
                    for action in civilian_actions:
                        if action in decision_line.upper():
                            decision = action.lower()
                            break
            
            if not decision:
                print(f"Warning: Could not extract decision from {model} response")
                decision = "unknown"
            
            return {
                "decision": decision,
                "reasoning": response_text,
                "success": True
            }
            
        except Exception as e:
            print(f"Error with {model}: {str(e)}")
            return {
                "decision": "error",
                "reasoning": f"Error: {str(e)}",
                "success": False
            }

    def run_experiment(self):
        """
        Run the complete one-shot experiment
        """
        condition_number = 0
        total_conditions = len(self.models) * 2 * len(self.scenarios) * self.runs_per_condition
        
        print(f"\nStarting One-Shot Experiment")
        print(f"Total conditions: {total_conditions}")
        if self.resume_from_condition > 0:
            print(f"Resuming from condition {self.resume_from_condition + 1}")
        print("="*60)
        
        for model in self.models:
            for frame in ["loss_frame", "gain_frame"]:
                for scenario_name in self.scenarios.keys():
                    for run in range(1, self.runs_per_condition + 1):
                        condition_number += 1
                        
                        # Skip conditions that have already been completed
                        if condition_number <= self.resume_from_condition:
                            continue
                        
                        print(f"\nCondition {condition_number}/{total_conditions}: {model}({frame}) - {scenario_name} (run {run})")
                        
                        # Get decision
                        result = self.make_decision(model, scenario_name, frame, run)
                        
                        # Record result
                        condition_result = {
                            "condition_number": condition_number,
                            "model": model,
                            "frame": frame,
                            "scenario": scenario_name,
                            "run": run,
                            "decision": result["decision"],
                            "reasoning": result["reasoning"],
                            "success": result["success"],
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        self.results.append(condition_result)
                        self.intermediate_results.append(condition_result)
                        
                        print(f"  Decision: {result['decision']}")
                        
                        # Save intermediate results every 15 conditions 
                        if condition_number % 15 == 0:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            intermediate_filename = f"civilian_intermediate_results_{condition_number}_{timestamp}.json"
                            with open(intermediate_filename, 'w') as f:
                                json.dump(self.results, f, indent=2)  # Save all results, not just intermediate
                            print(f"  Intermediate results saved to {intermediate_filename}")
                            self.intermediate_results = []
        
        # Save final results
        self.save_results()
        print(f"\nExperiment complete! Results saved.")

    def save_results(self):
        """
        Save final results in multiple formats
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON results
        json_filename = f"civilian_prospect_theory_{timestamp}.json"
        output_data = {
            "experiment_config": self.experiment_config,
            "results": self.results
        }
        
        with open(json_filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # CSV results
        csv_filename = f"civilian_prospect_theory_{timestamp}.csv"
        with open(csv_filename, 'w', newline='') as f:
            if self.results:
                fieldnames = self.results[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
        
        # Analysis summary
        self.generate_summary_analysis()
        
        print(f"Results saved to {json_filename} and {csv_filename}")

    def generate_summary_analysis(self):
        """
        Generate basic analysis of results
        """
        print("\n" + "="*60)
        print("PRELIMINARY ANALYSIS")
        print("="*60)
        
        # Count decisions by model and frame
        for model in self.models:
            model_results = [r for r in self.results if r['model'] == model and r['success']]
            
            print(f"\n{model.upper()}:")
            
            for frame in ["loss_frame", "gain_frame"]:
                frame_results = [r for r in model_results if r['frame'] == frame]
                if frame_results:
                    # Count actions for each option
                    option_counts = {}
                    for r in frame_results:
                        option_name = r['decision']
                        if option_name not in option_counts:
                            option_counts[option_name] = 0
                        option_counts[option_name] += 1
                    
                    # Sort options by count descending
                    sorted_options = sorted(option_counts.items(), key=lambda item: item[1], reverse=True)
                    
                    print(f"  {frame.replace('_frame', '').upper()}:")
                    for option_name, count in sorted_options:
                        print(f"    {option_name.upper()}: {count} times")
            
            # Calculate framing effect (risk-seeking behavior - third option is typically highest risk)
            loss_results = [r for r in model_results if r['frame'] == 'loss_frame']
            gain_results = [r for r in model_results if r['frame'] == 'gain_frame']
            
            if loss_results and gain_results:
                # Get all unique decisions to identify risk levels
                all_decisions = set([r['decision'] for r in model_results])
                all_decisions.discard('unknown')
                all_decisions.discard('error')
                
                if len(all_decisions) >= 3:
                    # Identify risky options (typically the latter options are riskier)
                    sorted_decisions = sorted(list(all_decisions))
                    risky_options = sorted_decisions[-2:]  # Last two options are typically riskier
                    
                    loss_risky = len([r for r in loss_results if r['decision'] in risky_options]) / len(loss_results)
                    gain_risky = len([r for r in gain_results if r['decision'] in risky_options]) / len(gain_results)
                    framing_effect = loss_risky - gain_risky
                    print(f"  Framing effect (risk-seeking in loss vs gain): {framing_effect:+.3f}")
                else:
                    print(f"  Insufficient decision variety for framing effect calculation")

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    force_fresh = "--fresh" in sys.argv or "--restart" in sys.argv
    
    if force_fresh:
        print("🚀 FRESH START requested via command line")
    
    # Run the one-shot prospect theory experiment
    experiment = OneShootProspectTheoryExperiment(
        models=["o3-2025-04-16", "claude-sonnet-4", "claude-sonnet-4-thinking", "gpt-4o", "gemini-1.5-pro"],
        runs_per_condition=25,
        experiment_version="one_shot_v1",
        force_fresh_start=force_fresh
    )
    
    print("💡 Tip: Use 'python run1_one_shot_prospect_theory.py --fresh' to force a fresh start")
    print("💡 Tip: Or delete run1_intermediate_results_*.json files to reset manually")
    print()
    
    experiment.run_experiment() 