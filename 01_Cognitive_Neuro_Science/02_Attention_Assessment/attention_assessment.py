import pygame
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import cv2
from sklearn.metrics import classification_report
from scipy import stats

# Configuration and Data Structures
@dataclass
class TestResult:
    """Data structure for storing test results"""
    participant_id: str
    test_type: str
    timestamp: datetime
    accuracy: float
    mean_rt: float
    rt_variability: float
    errors: Dict[str, int]
    raw_data: List[Dict]
    session_id: str

@dataclass
class ParticipantInfo:
    """Participant information structure"""
    id: str
    age: int
    gender: str
    education_level: str
    handedness: str
    medical_history: List[str]
    created_date: datetime

class AttentionAssessment:
    """
    Comprehensive Attention and Focus Assessment Platform
    
    This class provides a complete suite of cognitive tests for evaluating
    different types of attention including selective, divided, and sustained attention.
    """
    
    def __init__(self, screen_size: Tuple[int, int] = (1200, 800)):
        """Initialize the assessment platform"""
        pygame.init()
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("Attention Assessment Platform")
        
        # Colors
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'gray': (128, 128, 128),
            'yellow': (255, 255, 0)
        }
        
        # Fonts
        self.fonts = {
            'large': pygame.font.Font(None, 48),
            'medium': pygame.font.Font(None, 32),
            'small': pygame.font.Font(None, 24)
        }
        
        # Data storage
        self.results = []
        self.current_participant = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        os.makedirs("data/participants", exist_ok=True)
        os.makedirs("data/results", exist_ok=True)
        os.makedirs("data/reports", exist_ok=True)
    
    def set_participant(self, participant_info: ParticipantInfo):
        """Set current participant information"""
        self.current_participant = participant_info
        participant_dir = f"data/participants/{participant_info.id}"
        os.makedirs(participant_dir, exist_ok=True)
        
        # Save participant info
        with open(f"{participant_dir}/info.json", 'w') as f:
            json.dump(asdict(participant_info), f, default=str, indent=2)
    
    def show_instructions(self, instructions: str, title: str = "Instructions"):
        """Display test instructions"""
        self.screen.fill(self.colors['white'])
        
        # Title
        title_surface = self.fonts['large'].render(title, True, self.colors['black'])
        title_rect = title_surface.get_rect(center=(self.screen_size[0]//2, 100))
        self.screen.blit(title_surface, title_rect)
        
        # Instructions
        y_offset = 200
        lines = instructions.split('\n')
        for line in lines:
            text_surface = self.fonts['medium'].render(line, True, self.colors['black'])
            text_rect = text_surface.get_rect(center=(self.screen_size[0]//2, y_offset))
            self.screen.blit(text_surface, text_rect)
            y_offset += 40
        
        # Continue prompt
        prompt = self.fonts['medium'].render("Press SPACE to continue", True, self.colors['blue'])
        prompt_rect = prompt.get_rect(center=(self.screen_size[0]//2, self.screen_size[1] - 100))
        self.screen.blit(prompt, prompt_rect)
        
        pygame.display.flip()
        
        # Wait for space key
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
        return True
    
    def flanker_task(self, duration: int = 300, n_trials: int = 100) -> TestResult:
        """
        Flanker Task for Selective Attention Assessment
        
        Measures ability to suppress irrelevant information by responding to
        a central arrow while ignoring flanking arrows.
        
        Args:
            duration: Test duration in seconds
            n_trials: Number of trials to run
            
        Returns:
            TestResult object with performance metrics
        """
        instructions = """FLANKER TASK - SELECTIVE ATTENTION
        
You will see 5 arrows on the screen.
Focus on the MIDDLE arrow only.
        
Press LEFT arrow key if middle arrow points LEFT (←)
Press RIGHT arrow key if middle arrow points RIGHT (→)
        
Ignore the arrows on the sides - they may be distracting!
        
Respond as quickly and accurately as possible."""
        
        if not self.show_instructions(instructions, "Flanker Task"):
            return None
        
        # Trial configurations
        congruent_trials = [
            "<<<<<", ">>>>>",  # All arrows same direction
        ]
        incongruent_trials = [
            "<<><<", ">><<<",  # Middle arrow different direction
        ]
        
        # Generate trial sequence
        trials = []
        for _ in range(n_trials // 2):
            trials.extend([
                {'stimulus': random.choice(congruent_trials), 'condition': 'congruent'},
                {'stimulus': random.choice(incongruent_trials), 'condition': 'incongruent'}
            ])
        random.shuffle(trials)
        
        # Test execution
        results_data = []
        start_time = time.time()
        trial_count = 0
        
        clock = pygame.time.Clock()
        
        for trial in trials:
            if time.time() - start_time > duration:
                break
            
            trial_count += 1
            
            # Fixation cross
            self.screen.fill(self.colors['white'])
            self.draw_fixation_cross()
            pygame.display.flip()
            time.sleep(0.5)
            
            # Present stimulus
            self.screen.fill(self.colors['white'])
            stimulus_text = self.fonts['large'].render(trial['stimulus'], True, self.colors['black'])
            stimulus_rect = stimulus_text.get_rect(center=(self.screen_size[0]//2, self.screen_size[1]//2))
            self.screen.blit(stimulus_text, stimulus_rect)
            
            # Trial counter
            counter_text = self.fonts['small'].render(f"Trial {trial_count}/{len(trials)}", True, self.colors['gray'])
            self.screen.blit(counter_text, (10, 10))
            
            pygame.display.flip()
            
            # Get response
            trial_start = time.time()
            response = None
            response_time = None
            
            waiting_for_response = True
            while waiting_for_response:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None
                    if event.type == pygame.KEYDOWN:
                        if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                            response_time = time.time() - trial_start
                            response = 'left' if event.key == pygame.K_LEFT else 'right'
                            waiting_for_response = False
                
                # Timeout after 5 seconds
                if time.time() - trial_start > 5.0:
                    response = 'timeout'
                    response_time = 5.0
                    waiting_for_response = False
            
            # Determine accuracy
            correct_response = 'present' if trial['target_present'] else 'absent'
            accuracy = 1 if response == correct_response else 0
            
            # Record trial data
            trial_data = {
                'trial': trial_idx + 1,
                'set_size': trial['set_size'],
                'target_present': trial['target_present'],
                'correct_response': correct_response,
                'participant_response': response,
                'reaction_time': response_time,
                'accuracy': accuracy,
                'timestamp': time.time()
            }
            results_data.append(trial_data)
            
            # Brief inter-trial interval
            time.sleep(0.5)
        
        # Calculate performance metrics
        df = pd.DataFrame(results_data)
        
        # Overall metrics
        accuracy = df['accuracy'].mean() * 100
        correct_df = df[df['accuracy'] == 1]
        mean_rt = correct_df['reaction_time'].mean() * 1000 if len(correct_df) > 0 else 0
        rt_variability = correct_df['reaction_time'].std() * 1000 if len(correct_df) > 1 else 0
        
        # Search efficiency analysis
        search_slopes = {}
        for target_condition in ['present', 'absent']:
            condition_df = df[(df['target_present'] == (target_condition == 'present')) & (df['accuracy'] == 1)]
            if len(condition_df) > 0:
                # Calculate slope (ms per item)
                rt_by_size = condition_df.groupby('set_size')['reaction_time'].mean() * 1000
                set_sizes_used = sorted(condition_df['set_size'].unique())
                if len(set_sizes_used) > 1:
                    slope = np.polyfit(set_sizes_used, [rt_by_size[size] for size in set_sizes_used], 1)[0]
                    search_slopes[target_condition] = slope
        
        errors = {
            'false_alarms': len(df[(df['target_present'] == False) & (df['participant_response'] == 'present')]),
            'misses': len(df[(df['target_present'] == True) & (df['participant_response'] == 'absent')]),
            'timeouts': len(df[df['participant_response'] == 'timeout'])
        }
        
        # Create result object
        result = TestResult(
            participant_id=self.current_participant.id if self.current_participant else "unknown",
            test_type="visual_search_task",
            timestamp=datetime.now(),
            accuracy=accuracy,
            mean_rt=mean_rt,
            rt_variability=rt_variability,
            errors=errors,
            raw_data=results_data,
            session_id=self.session_id
        )
        
        # Display results summary
        extra_metrics = {}
        if 'present' in search_slopes:
            extra_metrics['Target Present Slope'] = f"{search_slopes['present']:.1f} ms/item"
        if 'absent' in search_slopes:
            extra_metrics['Target Absent Slope'] = f"{search_slopes['absent']:.1f} ms/item"
        
        self.show_results_summary(result, extra_metrics)
        
        self.results.append(result)
        return result
    
    def dual_nback_task(self, n_levels: int = 3, trials_per_level: int = 20) -> TestResult:
        """
        Dual N-Back Task for Working Memory and Divided Attention
        
        Assesses working memory capacity and divided attention by requiring
        participants to monitor multiple stimulus streams simultaneously.
        
        Args:
            n_levels: Maximum n-back level to test
            trials_per_level: Number of trials per level
            
        Returns:
            TestResult object with performance metrics
        """
        instructions = """DUAL N-BACK TASK - WORKING MEMORY & DIVIDED ATTENTION
        
You will see squares appearing in different positions AND hear letters.
        
For each trial, decide:
- Does the POSITION match the position N trials back? (Press 'P')
- Does the LETTER match the letter N trials back? (Press 'L')
- Both match? Press BOTH keys
- Neither matches? Don't press anything
        
We'll start with 1-back (remember 1 trial back).
Focus on both position AND sound simultaneously!"""
        
        if not self.show_instructions(instructions, "Dual N-Back Task"):
            return None
        
        # Available positions and letters
        positions = [(300, 200), (500, 200), (700, 200),
                    (300, 400), (500, 400), (700, 400),
                    (300, 600), (500, 600), (700, 600)]
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        
        results_data = []
        
        for n_back_level in range(1, n_levels + 1):
            # Show level instructions
            level_instructions = f"Starting {n_back_level}-Back Level\n\nRemember positions and letters from {n_back_level} trial(s) ago."
            if not self.show_instructions(level_instructions, f"{n_back_level}-Back Level"):
                return None
            
            # Generate trial sequence
            trial_sequence = []
            for trial_idx in range(trials_per_level + n_back_level):  # Extra trials for n-back comparison
                position = random.choice(positions)
                letter = random.choice(letters)
                
                # Determine matches for trials where comparison is possible
                position_match = False
                letter_match = False
                
                if trial_idx >= n_back_level:
                    comparison_trial = trial_sequence[trial_idx - n_back_level]
                    
                    # Introduce matches with certain probability
                    if random.random() < 0.3:  # 30% chance of position match
                        position = comparison_trial['position']
                        position_match = True
                    
                    if random.random() < 0.3:  # 30% chance of letter match
                        letter = comparison_trial['letter']
                        letter_match = True
                
                trial_sequence.append({
                    'trial': trial_idx,
                    'position': position,
                    'letter': letter,
                    'position_match': position_match,
                    'letter_match': letter_match,
                    'n_back_level': n_back_level
                })
            
            # Run trials for this level
            for trial in trial_sequence:
                if trial['trial'] < n_back_level:
                    continue  # Skip first n trials (no comparison possible)
                
                # Present stimulus
                self.screen.fill(self.colors['white'])
                
                # Level indicator
                level_text = self.fonts['medium'].render(f"{n_back_level}-Back Level", True, self.colors['blue'])
                self.screen.blit(level_text, (10, 10))
                
                # Trial counter
                actual_trial = trial['trial'] - n_back_level + 1
                counter_text = self.fonts['small'].render(f"Trial {actual_trial}/{trials_per_level}", True, self.colors['gray'])
                self.screen.blit(counter_text, (10, 50))
                
                # Position stimulus (square)
                pygame.draw.rect(self.screen, self.colors['blue'], 
                               (*trial['position'], 60, 60))
                
                # Instructions reminder
                reminder = self.fonts['small'].render("P = Position match, L = Letter match", True, self.colors['gray'])
                reminder_rect = reminder.get_rect(centerx=self.screen_size[0]//2, y=self.screen_size[1] - 30)
                self.screen.blit(reminder_text, reminder_rect)
                
                pygame.display.flip()
                
                # "Play" letter sound (simulate with text display)
                # In a real implementation, you'd use pygame.mixer for audio
                letter_text = self.fonts['large'].render(trial['letter'], True, self.colors['red'])
                letter_rect = letter_text.get_rect(center=(self.screen_size[0]//2, 100))
                self.screen.blit(letter_text, letter_rect)
                pygame.display.flip()
                
                # Get responses
                trial_start = time.time()
                position_response = False
                letter_response = False
                response_times = []
                
                stimulus_duration = 2.0  # 2 seconds to respond
                
                while time.time() - trial_start < stimulus_duration:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return None
                        if event.type == pygame.KEYDOWN:
                            response_time = time.time() - trial_start
                            if event.key == pygame.K_p:
                                position_response = True
                                response_times.append(response_time)
                            elif event.key == pygame.K_l:
                                letter_response = True
                                response_times.append(response_time)
                
                # Calculate accuracy
                position_correct = (position_response == trial['position_match'])
                letter_correct = (letter_response == trial['letter_match'])
                overall_correct = position_correct and letter_correct
                
                # Record trial data
                trial_data = {
                    'n_back_level': n_back_level,
                    'trial': actual_trial,
                    'position': trial['position'],
                    'letter': trial['letter'],
                    'position_match_actual': trial['position_match'],
                    'letter_match_actual': trial['letter_match'],
                    'position_response': position_response,
                    'letter_response': letter_response,
                    'position_correct': position_correct,
                    'letter_correct': letter_correct,
                    'overall_correct': overall_correct,
                    'response_times': response_times,
                    'fastest_rt': min(response_times) if response_times else None,
                    'timestamp': time.time()
                }
                results_data.append(trial_data)
                
                # Inter-trial interval
                time.sleep(0.8)
        
        # Calculate performance metrics
        df = pd.DataFrame(results_data)
        
        # Overall metrics
        overall_accuracy = df['overall_correct'].mean() * 100
        position_accuracy = df['position_correct'].mean() * 100
        letter_accuracy = df['letter_correct'].mean() * 100
        
        # Reaction time analysis
        rt_data = [rt for rts in df['response_times'] for rt in rts]
        mean_rt = np.mean(rt_data) * 1000 if rt_data else 0
        rt_variability = np.std(rt_data) * 1000 if len(rt_data) > 1 else 0
        
        # Performance by n-back level
        level_performance = df.groupby('n_back_level').agg({
            'overall_correct': 'mean',
            'position_correct': 'mean',
            'letter_correct': 'mean'
        }) * 100
        
        errors = {
            'total_errors': len(df[df['overall_correct'] == False]),
            'position_errors': len(df[df['position_correct'] == False]),
            'letter_errors': len(df[df['letter_correct'] == False])
        }
        
        # Create result object
        result = TestResult(
            participant_id=self.current_participant.id if self.current_participant else "unknown",
            test_type="dual_nback_task",
            timestamp=datetime.now(),
            accuracy=overall_accuracy,
            mean_rt=mean_rt,
            rt_variability=rt_variability,
            errors=errors,
            raw_data=results_data,
            session_id=self.session_id
        )
        
        # Display results summary
        extra_metrics = {
            'Position Accuracy': f"{position_accuracy:.1f}%",
            'Letter Accuracy': f"{letter_accuracy:.1f}%"
        }
        
        # Add level-specific performance
        for level in level_performance.index:
            extra_metrics[f'{level}-Back Accuracy'] = f"{level_performance.loc[level, 'overall_correct']:.1f}%"
        
        self.show_results_summary(result, extra_metrics)
        
        self.results.append(result)
        return result
    
    def generate_search_display_positions(self, n_items: int) -> List[Tuple[int, int]]:
        """Generate random positions for visual search display"""
        positions = []
        min_distance = 80  # Minimum distance between items
        
        while len(positions) < n_items:
            x = random.randint(100, self.screen_size[0] - 100)
            y = random.randint(150, self.screen_size[1] - 150)
            
            # Check distance from existing positions
            valid_position = True
            for existing_pos in positions:
                distance = np.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                if distance < min_distance:
                    valid_position = False
                    break
            
            if valid_position:
                positions.append((x, y))
        
        return positions
    
    def draw_fixation_cross(self, size: int = 20):
        """Draw a fixation cross at screen center"""
        center_x, center_y = self.screen_size[0] // 2, self.screen_size[1] // 2
        
        # Horizontal line
        pygame.draw.line(self.screen, self.colors['black'], 
                        (center_x - size, center_y), 
                        (center_x + size, center_y), 3)
        
        # Vertical line
        pygame.draw.line(self.screen, self.colors['black'], 
                        (center_x, center_y - size), 
                        (center_x, center_y + size), 3)
    
    def show_results_summary(self, result: TestResult, extra_metrics: Dict[str, str] = None):
        """Display test results summary"""
        self.screen.fill(self.colors['white'])
        
        # Title
        title = f"Test Results: {result.test_type.replace('_', ' ').title()}"
        title_surface = self.fonts['large'].render(title, True, self.colors['black'])
        title_rect = title_surface.get_rect(center=(self.screen_size[0]//2, 100))
        self.screen.blit(title_surface, title_rect)
        
        # Main metrics
        y_offset = 200
        metrics = [
            f"Overall Accuracy: {result.accuracy:.1f}%",
            f"Mean Reaction Time: {result.mean_rt:.1f} ms",
            f"RT Variability: {result.rt_variability:.1f} ms",
            f"Total Errors: {sum(result.errors.values()) if isinstance(result.errors, dict) else result.errors}"
        ]
        
        # Add extra metrics
        if extra_metrics:
            metrics.extend([f"{key}: {value}" for key, value in extra_metrics.items()])
        
        for metric in metrics:
            text_surface = self.fonts['medium'].render(metric, True, self.colors['black'])
            text_rect = text_surface.get_rect(center=(self.screen_size[0]//2, y_offset))
            self.screen.blit(text_surface, text_rect)
            y_offset += 40
        
        # Continue prompt
        prompt = self.fonts['medium'].render("Press SPACE to continue", True, self.colors['blue'])
        prompt_rect = prompt.get_rect(center=(self.screen_size[0]//2, self.screen_size[1] - 100))
        self.screen.blit(prompt, prompt_rect)
        
        pygame.display.flip()
        
        # Wait for space key
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
    
    def generate_comprehensive_report(self, participant_id: str = None) -> str:
        """Generate a comprehensive assessment report"""
        if participant_id:
            participant_results = [r for r in self.results if r.participant_id == participant_id]
        else:
            participant_results = self.results
        
        if not participant_results:
            return "No results available for report generation."
        
        # Create report
        report = []
        report.append("COMPREHENSIVE ATTENTION ASSESSMENT REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Participant ID: {participant_results[0].participant_id}")
        report.append(f"Session ID: {participant_results[0].session_id}")
        report.append("")
        
        # Summary table
        report.append("SUMMARY OF RESULTS")
        report.append("-" * 30)
        report.append(f"{'Test':<25} {'Accuracy':<12} {'Mean RT':<12} {'Errors'}")
        report.append("-" * 60)
        
        for result in participant_results:
            test_name = result.test_type.replace('_', ' ').title()[:24]
            accuracy = f"{result.accuracy:.1f}%"
            mean_rt = f"{result.mean_rt:.0f}ms"
            total_errors = sum(result.errors.values()) if isinstance(result.errors, dict) else result.errors
            
            report.append(f"{test_name:<25} {accuracy:<12} {mean_rt:<12} {total_errors}")
        
        report.append("")
        
        # Detailed analysis for each test
        for result in participant_results:
            report.append(f"DETAILED ANALYSIS: {result.test_type.replace('_', ' ').upper()}")
            report.append("-" * 40)
            report.append(f"Test Date: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Overall Accuracy: {result.accuracy:.1f}%")
            report.append(f"Mean Reaction Time: {result.mean_rt:.1f} ms")
            report.append(f"RT Variability (SD): {result.rt_variability:.1f} ms")
            report.append("")
            
            # Test-specific analysis
            if result.test_type == "flanker_task":
                report.append("Flanker Effect Analysis:")
                df = pd.DataFrame(result.raw_data)
                congruent_rt = df[df['condition'] == 'congruent']['reaction_time'].mean() * 1000
                incongruent_rt = df[df['condition'] == 'incongruent']['reaction_time'].mean() * 1000
                flanker_effect = incongruent_rt - congruent_rt
                report.append(f"  Flanker Effect: {flanker_effect:.1f} ms")
                report.append(f"  Congruent RT: {congruent_rt:.1f} ms")
                report.append(f"  Incongruent RT: {incongruent_rt:.1f} ms")
                
            elif result.test_type == "continuous_performance_test":
                report.append("Signal Detection Analysis:")
                hits = result.errors['hits']
                misses = result.errors['misses']
                false_alarms = result.errors['false_alarms']
                correct_rejections = result.errors['correct_rejections']
                
                hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
                false_alarm_rate = false_alarms / (false_alarms + correct_rejections) if (false_alarms + correct_rejections) > 0 else 0
                
                report.append(f"  Hit Rate: {hit_rate:.2f}")
                report.append(f"  False Alarm Rate: {false_alarm_rate:.2f}")
                report.append(f"  Hits: {hits}, Misses: {misses}")
                report.append(f"  False Alarms: {false_alarms}, Correct Rejections: {correct_rejections}")
            
            report.append("")
        
        # Clinical interpretation
        report.append("CLINICAL INTERPRETATION")
        report.append("-" * 30)
        
        # Calculate composite scores
        mean_accuracy = np.mean([r.accuracy for r in participant_results])
        mean_rt = np.mean([r.mean_rt for r in participant_results])
        
        if mean_accuracy >= 85:
            report.append("• Overall attention performance: NORMAL")
        elif mean_accuracy >= 70:
            report.append("• Overall attention performance: MILD IMPAIRMENT")
        else:
            report.append("• Overall attention performance: SIGNIFICANT IMPAIRMENT")
        
        if mean_rt <= 500:
            report.append("• Processing speed: NORMAL")
        elif mean_rt <= 750:
            report.append("• Processing speed: MILD SLOWING")
        else:
            report.append("• Processing speed: SIGNIFICANT SLOWING")
        
        report.append("")
        report.append("RECOMMENDATIONS")
        report.append("-" * 20)
        
        if mean_accuracy < 80:
            report.append("• Consider further neuropsychological evaluation")
            report.append("• Cognitive rehabilitation may be beneficial")
        
        if mean_rt > 600:
            report.append("• Evaluate for processing speed difficulties")
            report.append("• Consider accommodations for timed tasks")
        
        report.append("")
        report.append("Note: This automated report is for screening purposes only.")
        report.append("Clinical decisions should involve qualified professionals.")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = None):
        """Save all results to file"""
        if not filename:
            filename = f"attention_assessment_results_{self.session_id}.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            result_dict = asdict(result)
            result_dict['timestamp'] = result.timestamp.isoformat()
            serializable_results.append(result_dict)
        
        # Save to file
        filepath = f"data/results/{filename}"
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def export_to_csv(self, filename: str = None):
        """Export results to CSV format"""
        if not self.results:
            print("No results to export")
            return
        
        if not filename:
            filename = f"attention_results_{self.session_id}.csv"
        
        # Flatten results for CSV export
        csv_data = []
        for result in self.results:
            base_data = {
                'participant_id': result.participant_id,
                'test_type': result.test_type,
                'timestamp': result.timestamp,
                'session_id': result.session_id,
                'overall_accuracy': result.accuracy,
                'mean_rt': result.mean_rt,
                'rt_variability': result.rt_variability
            }
            
            # Add error data
            if isinstance(result.errors, dict):
                for error_type, count in result.errors.items():
                    base_data[f'error_{error_type}'] = count
            
            # Add trial-by-trial data
            for trial_data in result.raw_data:
                row_data = {**base_data, **trial_data}
                csv_data.append(row_data)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        filepath = f"data/results/{filename}"
        df.to_csv(filepath, index=False)
        print(f"Data exported to {filepath}")
    
    def quit(self):
        """Clean up and quit"""
        pygame.quit()

# Example usage and demo functions
def create_sample_participant() -> ParticipantInfo:
    """Create a sample participant for testing"""
    return ParticipantInfo(
        id="DEMO_001",
        age=25,
        gender="Other",
        education_level="Graduate",
        handedness="Right",
        medical_history=[],
        created_date=datetime.now()
    )

def run_demo():
    """Run a demonstration of the platform"""
    # Initialize assessment platform
    assessment = AttentionAssessment()
    
    # Create sample participant
    participant = create_sample_participant()
    assessment.set_participant(participant)
    
    print("Starting Attention Assessment Platform Demo")
    print("=" * 50)
    
    # Menu system
    while True:
        print("\nAvailable Tests:")
        print("1. Flanker Task (Selective Attention)")
        print("2. Continuous Performance Test (Sustained Attention)")
        print("3. Visual Search Task (Selective Attention)")
        print("4. Dual N-Back Task (Working Memory/Divided Attention)")
        print("5. Generate Report")
        print("6. Save Results")
        print("7. Export to CSV")
        print("8. Quit")
        
        choice = input("\nSelect test (1-8): ").strip()
        
        try:
            if choice == '1':
                print("Running Flanker Task...")
                result = assessment.flanker_task(duration=180, n_trials=50)  # 3 min, 50 trials
                if result:
                    print(f"Flanker Task completed. Accuracy: {result.accuracy:.1f}%")
                
            elif choice == '2':
                print("Running Continuous Performance Test...")
                result = assessment.continuous_performance_test(duration=300, target_frequency=0.3)  # 5 min
                if result:
                    print(f"CPT completed. Accuracy: {result.accuracy:.1f}%")
                
            elif choice == '3':
                print("Running Visual Search Task...")
                result = assessment.visual_search_task(n_trials=30, set_sizes=[4, 8, 12])  # 30 trials per set size
                if result:
                    print(f"Visual Search completed. Accuracy: {result.accuracy:.1f}%")
                
            elif choice == '4':
                print("Running Dual N-Back Task...")
                result = assessment.dual_nback_task(n_levels=3, trials_per_level=15)  # 3 levels, 15 trials each
                if result:
                    print(f"Dual N-Back completed. Accuracy: {result.accuracy:.1f}%")
                
            elif choice == '5':
                if assessment.results:
                    report = assessment.generate_comprehensive_report()
                    print("\n" + report)
                    
                    # Save report
                    report_filename = f"report_{participant.id}_{assessment.session_id}.txt"
                    with open(f"data/reports/{report_filename}", 'w') as f:
                        f.write(report)
                    print(f"\nReport saved to data/reports/{report_filename}")
                else:
                    print("No results available. Complete some tests first.")
                
            elif choice == '6':
                assessment.save_results()
                
            elif choice == '7':
                assessment.export_to_csv()
                
            elif choice == '8':
                print("Thank you for using the Attention Assessment Platform!")
                break
                
            else:
                print("Invalid choice. Please select 1-8.")
                
        except KeyboardInterrupt:
            print("\nTest interrupted by user.")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    assessment.quit()

if __name__ == "__main__":
    run_demo() Timeout after 2 seconds
                if time.time() - trial_start > 2.0:
                    response = 'timeout'
                    response_time = 2.0
                    waiting_for_response = False
            
            # Determine correct response
            middle_arrow = trial['stimulus'][2]
            correct_response = 'left' if middle_arrow == '<' else 'right'
            
            # Record trial data
            trial_data = {
                'trial': trial_count,
                'stimulus': trial['stimulus'],
                'condition': trial['condition'],
                'correct_response': correct_response,
                'participant_response': response,
                'reaction_time': response_time,
                'accuracy': 1 if response == correct_response else 0,
                'timestamp': time.time()
            }
            results_data.append(trial_data)
            
            # Brief inter-trial interval
            time.sleep(0.2)
        
        # Calculate performance metrics
        df = pd.DataFrame(results_data)
        
        # Overall metrics
        accuracy = df['accuracy'].mean() * 100
        mean_rt = df[df['accuracy'] == 1]['reaction_time'].mean() * 1000  # Convert to ms
        rt_variability = df[df['accuracy'] == 1]['reaction_time'].std() * 1000
        
        # Condition-specific metrics
        congruent_df = df[df['condition'] == 'congruent']
        incongruent_df = df[df['condition'] == 'incongruent']
        
        congruent_accuracy = congruent_df['accuracy'].mean() * 100
        incongruent_accuracy = incongruent_df['accuracy'].mean() * 100
        
        congruent_rt = congruent_df[congruent_df['accuracy'] == 1]['reaction_time'].mean() * 1000
        incongruent_rt = incongruent_df[incongruent_df['accuracy'] == 1]['reaction_time'].mean() * 1000
        
        # Flanker effect (incongruent RT - congruent RT)
        flanker_effect = incongruent_rt - congruent_rt
        
        # Error analysis
        errors = {
            'total_errors': len(df[df['accuracy'] == 0]),
            'congruent_errors': len(congruent_df[congruent_df['accuracy'] == 0]),
            'incongruent_errors': len(incongruent_df[incongruent_df['accuracy'] == 0]),
            'timeouts': len(df[df['participant_response'] == 'timeout'])
        }
        
        # Create result object
        result = TestResult(
            participant_id=self.current_participant.id if self.current_participant else "unknown",
            test_type="flanker_task",
            timestamp=datetime.now(),
            accuracy=accuracy,
            mean_rt=mean_rt,
            rt_variability=rt_variability,
            errors=errors,
            raw_data=results_data,
            session_id=self.session_id
        )
        
        # Display results summary
        self.show_results_summary(result, {
            'Flanker Effect': f"{flanker_effect:.1f} ms",
            'Congruent Accuracy': f"{congruent_accuracy:.1f}%",
            'Incongruent Accuracy': f"{incongruent_accuracy:.1f}%",
            'Congruent RT': f"{congruent_rt:.1f} ms",
            'Incongruent RT': f"{incongruent_rt:.1f} ms"
        })
        
        self.results.append(result)
        return result
    
    def continuous_performance_test(self, duration: int = 600, target_frequency: float = 0.3) -> TestResult:
        """
        Continuous Performance Test for Sustained Attention
        
        Evaluates sustained attention by having participants respond to
        specific target stimuli in a stream of non-target stimuli.
        
        Args:
            duration: Test duration in seconds
            target_frequency: Proportion of trials that are targets
            
        Returns:
            TestResult object with performance metrics
        """
        instructions = """CONTINUOUS PERFORMANCE TEST - SUSTAINED ATTENTION
        
You will see a series of letters appearing one at a time.
        
Press SPACEBAR when you see the letter 'X'
DO NOT respond to any other letters
        
This test will continue for several minutes.
Stay alert and maintain focus throughout!"""
        
        if not self.show_instructions(instructions, "Continuous Performance Test"):
            return None
        
        # Generate stimulus sequence
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z']
        target_letter = 'X'
        
        # Calculate number of trials
        stimulus_duration = 0.5  # 500ms per stimulus
        isi = 1.0  # 1000ms inter-stimulus interval
        trial_duration = stimulus_duration + isi
        n_trials = int(duration / trial_duration)
        n_targets = int(n_trials * target_frequency)
        
        # Create trial sequence
        trials = ['non-target'] * (n_trials - n_targets) + ['target'] * n_targets
        random.shuffle(trials)
        
        # Test execution
        results_data = []
        start_time = time.time()
        
        clock = pygame.time.Clock()
        
        for trial_idx, trial_type in enumerate(trials):
            if time.time() - start_time > duration:
                break
            
            # Select stimulus
            if trial_type == 'target':
                stimulus = target_letter
            else:
                stimulus = random.choice(letters)
            
            # Present stimulus
            self.screen.fill(self.colors['white'])
            
            # Progress indicator
            progress = trial_idx / len(trials)
            progress_width = 400
            progress_rect = pygame.Rect((self.screen_size[0] - progress_width) // 2, 50, progress_width, 20)
            pygame.draw.rect(self.screen, self.colors['gray'], progress_rect)
            filled_width = int(progress * progress_width)
            filled_rect = pygame.Rect((self.screen_size[0] - progress_width) // 2, 50, filled_width, 20)
            pygame.draw.rect(self.screen, self.colors['blue'], filled_rect)
            
            # Stimulus
            stimulus_text = self.fonts['large'].render(stimulus, True, self.colors['black'])
            stimulus_rect = stimulus_text.get_rect(center=(self.screen_size[0]//2, self.screen_size[1]//2))
            self.screen.blit(stimulus_text, stimulus_rect)
            
            pygame.display.flip()
            
            # Record responses during stimulus presentation
            trial_start = time.time()
            responses = []
            
            while time.time() - trial_start < stimulus_duration:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            response_time = time.time() - trial_start
                            responses.append(response_time)
            
            # Inter-stimulus interval
            self.screen.fill(self.colors['white'])
            pygame.display.flip()
            
            # Continue recording responses during ISI
            while time.time() - trial_start < trial_duration:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            response_time = time.time() - trial_start
                            responses.append(response_time)
            
            # Analyze responses for this trial
            if trial_type == 'target':
                if responses:
                    # Hit - correct response to target
                    trial_outcome = 'hit'
                    reaction_time = min(responses)  # Use fastest response
                else:
                    # Miss - no response to target
                    trial_outcome = 'miss'
                    reaction_time = None
            else:
                if responses:
                    # False alarm - response to non-target
                    trial_outcome = 'false_alarm'
                    reaction_time = min(responses)
                else:
                    # Correct rejection - no response to non-target
                    trial_outcome = 'correct_rejection'
                    reaction_time = None
            
            # Record trial data
            trial_data = {
                'trial': trial_idx + 1,
                'stimulus': stimulus,
                'trial_type': trial_type,
                'responses': len(responses),
                'reaction_time': reaction_time,
                'outcome': trial_outcome,
                'timestamp': time.time()
            }
            results_data.append(trial_data)
        
        # Calculate performance metrics
        df = pd.DataFrame(results_data)
        
        # Signal detection theory metrics
        hits = len(df[(df['trial_type'] == 'target') & (df['outcome'] == 'hit')])
        misses = len(df[(df['trial_type'] == 'target') & (df['outcome'] == 'miss')])
        false_alarms = len(df[(df['trial_type'] == 'non-target') & (df['outcome'] == 'false_alarm')])
        correct_rejections = len(df[(df['trial_type'] == 'non-target') & (df['outcome'] == 'correct_rejection')])
        
        # Calculate rates
        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
        false_alarm_rate = false_alarms / (false_alarms + correct_rejections) if (false_alarms + correct_rejections) > 0 else 0
        
        # d-prime (sensitivity) and criterion
        z_hit = stats.norm.ppf(max(0.01, min(0.99, hit_rate)))
        z_fa = stats.norm.ppf(max(0.01, min(0.99, false_alarm_rate)))
        d_prime = z_hit - z_fa
        criterion = -(z_hit + z_fa) / 2
        
        # Reaction time analysis
        hit_rts = df[(df['outcome'] == 'hit') & (df['reaction_time'].notna())]['reaction_time']
        mean_rt = hit_rts.mean() * 1000 if len(hit_rts) > 0 else 0
        rt_variability = hit_rts.std() * 1000 if len(hit_rts) > 1 else 0
        
        # Overall accuracy
        accuracy = (hits + correct_rejections) / len(df) * 100
        
        errors = {
            'hits': hits,
            'misses': misses,
            'false_alarms': false_alarms,
            'correct_rejections': correct_rejections
        }
        
        # Create result object
        result = TestResult(
            participant_id=self.current_participant.id if self.current_participant else "unknown",
            test_type="continuous_performance_test",
            timestamp=datetime.now(),
            accuracy=accuracy,
            mean_rt=mean_rt,
            rt_variability=rt_variability,
            errors=errors,
            raw_data=results_data,
            session_id=self.session_id
        )
        
        # Display results summary
        self.show_results_summary(result, {
            "d' (Sensitivity)": f"{d_prime:.2f}",
            'Hit Rate': f"{hit_rate:.2f}",
            'False Alarm Rate': f"{false_alarm_rate:.2f}",
            'Criterion': f"{criterion:.2f}"
        })
        
        self.results.append(result)
        return result
    
    def visual_search_task(self, n_trials: int = 50, set_sizes: List[int] = [4, 8, 16]) -> TestResult:
        """
        Visual Search Task for Selective Attention
        
        Measures efficiency of selective attention by having participants
        locate target items among distractors.
        
        Args:
            n_trials: Number of trials per set size
            set_sizes: List of set sizes to test
            
        Returns:
            TestResult object with performance metrics
        """
        instructions = """VISUAL SEARCH TASK - SELECTIVE ATTENTION
        
You will see displays with colored shapes.
Look for a RED CIRCLE among the other shapes.
        
Press 'P' if the RED CIRCLE is PRESENT
Press 'A' if the RED CIRCLE is ABSENT
        
Respond as quickly and accurately as possible.
The number of shapes will vary between trials."""
        
        if not self.show_instructions(instructions, "Visual Search Task"):
            return None
        
        # Generate trials
        trials = []
        for set_size in set_sizes:
            for _ in range(n_trials):
                # Target present/absent
                target_present = random.choice([True, False])
                trials.append({
                    'set_size': set_size,
                    'target_present': target_present
                })
        
        random.shuffle(trials)
        
        # Test execution
        results_data = []
        
        for trial_idx, trial in enumerate(trials):
            # Generate display
            positions = self.generate_search_display_positions(trial['set_size'])
            
            # Create stimuli
            stimuli = []
            target_position = None
            
            if trial['target_present']:
                # Place target
                target_position = random.choice(positions)
                stimuli.append({
                    'position': target_position,
                    'color': self.colors['red'],
                    'shape': 'circle',
                    'is_target': True
                })
                positions.remove(target_position)
            
            # Place distractors
            distractor_colors = [self.colors['blue'], self.colors['green']]
            distractor_shapes = ['circle', 'square']
            
            for pos in positions:
                # Ensure distractors are not red circles
                if trial['target_present'] or random.random() < 0.5:
                    color = random.choice(distractor_colors)
                    shape = random.choice(distractor_shapes)
                    # Avoid red circles as distractors
                    while color == self.colors['red'] and shape == 'circle':
                        color = random.choice(distractor_colors)
                        shape = random.choice(distractor_shapes)
                else:
                    color = random.choice([self.colors['red']] + distractor_colors)
                    shape = 'square'  # Red squares as distractors
                
                stimuli.append({
                    'position': pos,
                    'color': color,
                    'shape': shape,
                    'is_target': False
                })
            
            # Present search display
            self.screen.fill(self.colors['white'])
            
            # Draw stimuli
            for stimulus in stimuli:
                x, y = stimulus['position']
                if stimulus['shape'] == 'circle':
                    pygame.draw.circle(self.screen, stimulus['color'], (x, y), 20)
                else:  # square
                    rect = pygame.Rect(x-20, y-20, 40, 40)
                    pygame.draw.rect(self.screen, stimulus['color'], rect)
            
            # Trial info
            info_text = self.fonts['small'].render(
                f"Trial {trial_idx + 1}/{len(trials)} | Set Size: {trial['set_size']}", 
                True, self.colors['gray']
            )
            self.screen.blit(info_text, (10, 10))
            
            # Instructions reminder
            reminder_text = self.fonts['small'].render("P = Present, A = Absent", True, self.colors['gray'])
            reminder_rect = reminder_text.get_rect(centerx=self.screen_size[0]//2, y=self.screen_size[1] - 30)
            self.screen.blit(reminder_text, reminder_rect)
            
            pygame.display.flip()
            
            # Get response
            trial_start = time.time()
            response = None
            response_time = None
            
            waiting_for_response = True
            while waiting_for_response:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None
                    if event.type == pygame.KEYDOWN:
                        if event.key in [pygame.K_p, pygame.K_a]:
                            response_time = time.time() - trial_start
                            response = 'present' if event.key == pygame.K_p else 'absent'
                            waiting_for_response = False
                
                #