# Attention and Focus Assessment Platform

## Overview

A comprehensive cognitive assessment platform designed to evaluate different types of attention (selective, divided, sustained) and track cognitive performance over time. This system provides interactive tests based on established neuropsychological paradigms and generates detailed analytics for clinical, educational, and research applications.

## Features

### 🧠 Cognitive Tests
- **Selective Attention**: Flanker Task, Visual Search, Stroop Test
- **Divided Attention**: Dual N-Back, Multi-Modal Tasks
- **Sustained Attention**: Continuous Performance Test (CPT), Vigilance Tasks
- **Working Memory**: Digit Span, Spatial Span Tests

### 📊 Analytics & Reporting
- Real-time performance metrics
- Progress tracking over time
- Detailed statistical analysis
- Comparative normative data
- Export capabilities (CSV, PDF reports)

### 🎯 Applications
- ADHD screening and monitoring
- Cognitive rehabilitation
- Educational assessment
- Research in attention disorders
- Performance optimization training

## Technical Stack

- **Python 3.8+**
- **Pygame** - Interactive test interface
- **OpenCV** - Computer vision for eye tracking (optional)
- **Pandas** - Data analysis and management
- **NumPy** - Numerical computations
- **Matplotlib/Seaborn** - Data visualization
- **Scikit-learn** - Statistical analysis
- **Tkinter** - GUI components

## Installation

### Prerequisites
```bash
pip install pygame opencv-python pandas numpy matplotlib seaborn scikit-learn pillow
```

### Clone Repository
```bash
git clone https://github.com/yourusername/attention-assessment-platform.git
cd attention-assessment-platform
```

### Setup
```bash
python setup.py
```

## Quick Start

```python
from attention_platform import AttentionAssessment

# Initialize assessment
assessment = AttentionAssessment()

# Run selective attention test
results = assessment.run_flanker_task(duration=300)  # 5 minutes

# View results
assessment.generate_report(results)
```

## Test Descriptions

### 1. Flanker Task (Selective Attention)
Measures ability to suppress irrelevant information. Participants respond to a central arrow while ignoring flanking arrows that may point in the same (congruent) or opposite (incongruent) direction.

### 2. Continuous Performance Test (Sustained Attention)
Evaluates sustained attention over extended periods. Participants monitor a stream of stimuli and respond to specific target sequences.

### 3. Dual N-Back (Divided Attention)
Assesses working memory and divided attention by requiring participants to monitor multiple stimulus streams simultaneously.

### 4. Visual Search Task
Measures selective attention efficiency by having participants locate target items among distractors in visual arrays.

## Data Structure

```
data/
├── participants/
│   ├── participant_001/
│   │   ├── sessions/
│   │   ├── results/
│   │   └── reports/
├── normative_data/
└── exports/
```

## Configuration

Edit `config.json` to customize:
- Test parameters
- Difficulty levels
- Session duration
- Output formats
- Normative data sources

## API Reference

### AttentionAssessment Class

#### Methods
- `run_flanker_task(duration, difficulty)` - Execute Flanker paradigm
- `run_cpt_task(duration, target_frequency)` - Continuous Performance Test
- `run_visual_search(trials, set_sizes)` - Visual search assessment
- `run_dual_nback(levels, trials_per_level)` - Dual N-Back test
- `generate_report(results, format='pdf')` - Create assessment report
- `track_progress(participant_id, timeframe)` - Monitor performance over time

#### Properties
- `current_session` - Active assessment session
- `participant_data` - Current participant information
- `test_results` - Latest test outcomes

## Research Applications

### Clinical Assessment
- ADHD diagnosis support
- Cognitive decline monitoring
- Treatment efficacy evaluation
- Rehabilitation progress tracking

### Educational Settings
- Learning disability identification
- Attention training programs
- Academic performance correlation
- Intervention effectiveness

### Research Studies
- Attention mechanism investigation
- Cognitive load assessment
- Individual differences analysis
- Neuroplasticity studies

## Performance Metrics

### Accuracy Measures
- Response accuracy percentage
- Error rates by condition
- Commission vs. omission errors

### Reaction Time Analysis
- Mean response time
- Response time variability
- Speed-accuracy tradeoffs

### Attention Indices
- Alertness index
- Orienting efficiency
- Executive control effectiveness
- Sustained attention stability

## Validation & Reliability

The platform implements validated paradigms from cognitive neuroscience literature:
- Flanker Task (Eriksen & Eriksen, 1974)
- Continuous Performance Test (Rosvold et al., 1956)
- Dual N-Back (Jaeggi et al., 2008)
- Visual Search (Treisman & Gelade, 1980)

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this platform in your research, please cite:

```bibtex
@software{attention_assessment_platform,
  title={Attention and Focus Assessment Platform},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/attention-assessment-platform}
}
```

## Contact

- **Email**: your.email@university.edu
- **LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)
- **Research Gate**: [Your Profile](https://researchgate.net/profile/yourprofile)

## Acknowledgments

- Cognitive neuroscience research community
- Open-source Python community
- Validation study participants
- Clinical collaborators

---

**Note**: This platform is for research and educational purposes. Clinical decisions should always involve qualified healthcare professionals.