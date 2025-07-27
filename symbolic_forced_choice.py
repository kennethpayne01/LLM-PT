#!/usr/bin/env python3
"""
FORCED CHOICE Symbolic Prospect Theory Test
Modified from corrected version to force models to choose one option
Tests whether framing effects emerge when models must break ties
"""

import json
import os
import time
from datetime import datetime
import asyncio
import re
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import csv
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class CorrectedSymbolicProspectTheoryTest:
    
    def __init__(self):
        # Streamlined to two key models: advanced reasoning vs efficient
        self.models = [
            "o3-2025-04-16",
            "gpt-4o"
        ]
        
        # 50 runs per condition for statistical power
        self.runs_per_condition = 50
        
        # Initialize API clients
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # TRUE prospect theory: same outcomes, different reference points
        self.prospect_scenarios = {
            "scenario_α": {
                # Loss prevention frame: starting below reference, preventing further losses
                "frame_L": {
                    "current_state": "s₀ = -3",
                    "reference_point": "R = 0", 
                    "outcome_space": "Ω = {-3, -2, -1, 0, +1}",
                    "transformation": "Δ(ωᵢ) = ωᵢ - s₀",  # Change from current state
                    "value_function": "V(Δ) = Δ",  # Direct evaluation of change
                    "frame_type": "loss_prevention"
                },
                # Gain achievement frame: starting at reference, achieving gains  
                "frame_G": {
                    "current_state": "s₀ = 0",
                    "reference_point": "R = 0",
                    "outcome_space": "Ω = {0, +1, +2, +3, +4}",  # Same CHANGES as frame_L
                    "transformation": "Δ(ωᵢ) = ωᵢ - s₀", 
                    "value_function": "U(Δ) = Δ",
                    "frame_type": "gain_achievement"
                }
            },
            
            "scenario_β": {
                "frame_L": {
                    "current_state": "s₀ = -2", 
                    "reference_point": "R = 0",
                    "outcome_space": "Ω = {-2, -1, 0, +1, +2}",
                    "transformation": "Δ(ωᵢ) = ωᵢ - s₀",
                    "value_function": "V(Δ) = Δ", 
                    "frame_type": "loss_prevention"
                },
                "frame_G": {
                    "current_state": "s₀ = 0",
                    "reference_point": "R = 0", 
                    "outcome_space": "Ω = {0, +1, +2, +3, +4}",
                    "transformation": "Δ(ωᵢ) = ωᵢ - s₀",
                    "value_function": "U(Δ) = Δ",
                    "frame_type": "gain_achievement"
                }
            }
        }
        
        # Actions with ONLY probability distributions - models must calculate expected values themselves!
        self.action_space = {
            "𝒜₁": {
                "change_distribution": "P(Δ=+0.5) = 1.0"
            },
            "𝒜₂": {
                "change_distribution": "P(Δ=0) = 0.5, P(Δ=+1) = 0.5"
            },
            "𝒜₃": {
                "change_distribution": "P(Δ=0) = 0.75, P(Δ=+2) = 0.25"
            }
        }
        
        self.results = []
    
    def generate_prospect_prompt(self, scenario_key, frame_type, model_name):
        """Generate true reference point dependent prompt - NO PRE-CALCULATED VALUES!"""
        
        scenario = self.prospect_scenarios[scenario_key][frame_type]
        
        prompt = f"""◇ REFERENCE POINT PROBLEM ◇

⊢ STATE:
  {scenario['current_state']}
  {scenario['reference_point']}  

⊢ EVALUATION:
  {scenario['transformation']}
  {scenario['value_function']}

⊢ ACTIONS:

⊢ 𝒜₁:
  {self.action_space['𝒜₁']['change_distribution']}

⊢ 𝒜₂:
  {self.action_space['𝒜₂']['change_distribution']}

⊢ 𝒜₃:
  {self.action_space['𝒜₃']['change_distribution']}

⊢ SOLUTION:
  Find 𝒶* = arg sup 𝔼[V(Δ(𝒶))]

Provide your complete mathematical analysis and select the single optimal action 𝒶*. If multiple actions are optimal, you must choose one."""
        
        return prompt

    async def query_model(self, model_name, prompt):
        """Query a specific model with the prompt"""
        
        try:
            if model_name == "o3-2025-04-16":
                response = self.openai_client.chat.completions.create(
                    model="o3-2025-04-16",
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=5000,
                    temperature=1  # Fixed for o3
                )
                return response.choices[0].message.content.strip()
                
            elif model_name == "gpt-4o":
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.1
                )
                return response.choices[0].message.content.strip()
                
        except Exception as e:
            print(f"Error querying {model_name}: {e}")
            return None
    
    def parse_decision(self, response):
        """Parse the final forced-choice decision from the model's response."""
        if not response:
            return None

        # More robust patterns to find the single chosen action.
        # The key is to find the explicit declaration of the final choice, e.g., "𝒶* = 𝒜₁".
        specific_patterns = [
            r'optimal action is\s*𝒜(\d)',
            r'optimal action:?\s*.*\s*𝒜(\d)',
            r'optimal choice:?\s*.*\s*𝒜(\d)',
            r'chosen optimal action:?\s*.*\s*𝒜(\d)',
            r'the single optimal action is\s*𝒜(\d)',
            r'final answer is\s*𝒜(\d)',
            r'answer is\s*𝒜(\d)',
            r'solution is\s*𝒜(\d)',
            r'choose\s*𝒜(\d)',
            r'select\s*𝒜(\d)',
            r'(?:a|𝒶)\*?\s*=\s*𝒜(\d)',  # Handles a* = A1, 𝒶* = A1, etc.
        ]

        # Search the entire response for any of these patterns.
        # We check in reverse order of specifity, with the most likely candidates first.
        for pattern in reversed(specific_patterns):
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            if matches:
                # Return the last match found, as it's most likely the final conclusion.
                action_num = matches[-1]
                return f"𝒜{action_num}"
        
        # If no specific choice is found, the parse has failed for this experiment's purpose.
        return "unclear"
    
    async def run_experiment(self):
        """Run the corrected mathematical reasoning experiment"""
        
        print("="*80)
        print("CORRECTED SYMBOLIC PROSPECT THEORY TEST")
        print("CRITICAL FIX: Models must calculate expected values themselves")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []
        condition_number = 0
        
        total_conditions = len(self.models) * len(self.prospect_scenarios) * 2 * self.runs_per_condition
        print(f"\nTotal conditions to test: {total_conditions}")
        
        for model_name in self.models:
            print(f"\n{'='*60}")
            print(f"TESTING MODEL: {model_name.upper()}")
            print(f"{'='*60}")
            
            for scenario_key in self.prospect_scenarios.keys():
                for frame_type in ["frame_L", "frame_G"]:
                    frame_name = "Loss Prevention" if frame_type == "frame_L" else "Gain Achievement"
                    
                    print(f"\n📊 {scenario_key} - {frame_name}")
                    print(f"   Running {self.runs_per_condition} iterations...")
                    
                    for run in range(1, self.runs_per_condition + 1):
                        condition_number += 1
                        
                        try:
                            prompt = self.generate_prospect_prompt(scenario_key, frame_type, model_name)
                            response = await self.query_model(model_name, prompt)
                            decision = self.parse_decision(response)
                            
                            result = {
                                "condition_number": condition_number,
                                "timestamp": timestamp,
                                "scenario": scenario_key,
                                "frame_type": frame_type,
                                "frame_name": frame_name,
                                "model": model_name,
                                "run": run,
                                "decision": decision,
                                "success": decision != "unclear",
                                "response": response
                            }
                            
                            self.results.append(result)
                            
                            # Progress indicator
                            if run % 10 == 0:
                                success_rate = len([r for r in self.results[-10:] if r['success']]) / 10
                                print(f"     Run {run}/{self.runs_per_condition} - Success rate: {success_rate:.1%}")
                            
                            # Rate limiting
                            await asyncio.sleep(0.5)
                            
                        except Exception as e:
                            print(f"     Error on run {run}: {e}")
                            # Record failed attempt
                            self.results.append({
                                "condition_number": condition_number,
                                "timestamp": timestamp,
                                "scenario": scenario_key,
                                "frame_type": frame_type,
                                "frame_name": frame_name,
                                "model": model_name,
                                "run": run,
                                "decision": "error",
                                "success": False,
                                "response": f"Error: {str(e)}"
                            })
                            
                            await asyncio.sleep(2)  # Longer delay after errors
                        
                        # Save intermediate results every 50 conditions
                        if condition_number % 50 == 0:
                            await self.save_intermediate_results(timestamp, condition_number)
        
        # Final save
        await self.save_results(timestamp)
        await self.generate_analysis()
    
    async def save_intermediate_results(self, timestamp, condition_number):
        """Save intermediate results"""
        filename = f"symbolic_forced_choice_intermediate_{condition_number}_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Intermediate results saved: {filename}")
    
    async def save_results(self, timestamp):
        """Save final results in both JSON and CSV formats"""
        
        # JSON format (detailed)
        json_filename = f"symbolic_forced_choice_results_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # CSV format (analysis-ready)
        csv_filename = f"symbolic_forced_choice_results_{timestamp}.csv"
        if self.results:
            with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
        
        print(f"\n💾 Final results saved:")
        print(f"   JSON: {json_filename}")
        print(f"   CSV: {csv_filename}")
    
    async def generate_analysis(self):
        """Generate analysis of corrected results"""
        print("\n" + "="*60)
        print("CORRECTED MATHEMATICAL REASONING ANALYSIS")
        print("="*60)
        
        successful_results = [r for r in self.results if r['success']]
        total_results = len(self.results)
        success_rate = len(successful_results) / total_results if total_results > 0 else 0
        
        print(f"\nOverall success rate: {success_rate:.1%} ({len(successful_results)}/{total_results})")
        
        for model in self.models:
            model_results = [r for r in successful_results if r['model'] == model]
            
            print(f"\n{model.upper()}:")
            
            if not model_results:
                print("  No successful results")
                continue
            
            for frame in ["frame_L", "frame_G"]:
                frame_results = [r for r in model_results if r['frame_type'] == frame]
                frame_label = "Loss Prevention" if frame == "frame_L" else "Gain Achievement"
                
                if frame_results:
                    # Count decision types
                    decisions = {}
                    for result in frame_results:
                        decision = result['decision']
                        decisions[decision] = decisions.get(decision, 0) + 1
                    
                    print(f"  {frame_label} ({len(frame_results)} decisions):")
                    for decision, count in sorted(decisions.items()):
                        percentage = count / len(frame_results) * 100
                        print(f"    {decision}: {count} ({percentage:.1f}%)")
        
        print(f"\n🎯 Key Question: Do models now show TRUE mathematical reasoning?")
        print(f"   - Models must calculate E[Δ] = 0.5 for all actions themselves")
        print(f"   - No pre-calculated values provided")
        print(f"   - Test of genuine mathematical vs linguistic reasoning")

# Main execution
async def main():
    experiment = CorrectedSymbolicProspectTheoryTest()
    await experiment.run_experiment()

if __name__ == "__main__":
    asyncio.run(main()) 