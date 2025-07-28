# TE_autism - Turing Experiment for Autism Community

## Project Overview

This project investigates whether AI-generated responses can be distinguished from human responses when answering questions from the autism community. The study compares three types of responses through a comprehensive evaluation framework:

- **In-group responses**: Answers from autism community members
- **Out-group responses**: Answers from general population 
- **AI-generated responses**: Responses from OpenAI models (GPT-4, GPT-3.5-turbo)

### Research Pipeline

1. **Data Collection & Demographics**: Load 20 autism community questions and 100 out-group human answers, then generate 100 unique demographic profiles across 5 dimensions (age, gender, location, autism experience, knowledge level)

2. **AI Response Simulation**: Use demographic-aware prompting to generate AI responses that incorporate specific background contexts, with cost optimization reducing API calls by 76%

3. **Multi-Rater Evaluation**: Implement a three-tier rating system with researchers, individuals with autism, and autism experts evaluating responses across 5 dimensions

4. **Statistical Analysis**: Calculate inter-rater reliability using Krippendorff's α and perform group comparisons using Mann-Whitney U tests

5. **Results & Insights**: Generate comprehensive reports comparing response quality and distinguishability across groups

## File Structure
```
TE_autism/
├── scripts/
│   └── TE_autism.ipynb          # Main analysis notebook
├── src/                         # Python modules
├── requirements.txt             # pip dependencies
├── environment.yml              # conda environment
└── README.md                    # Documentation
```

## Quick Start

### 1. Environment Setup
**Using pip:**
```bash
pip install -r requirements.txt
```

**Using conda:**
```bash
conda env create -f environment.yml
conda activate te_autism
```

### 2. API Configuration
Create in project root:
- `openai_api_key.txt` - Your OpenAI API key
- `openai_organization.txt` - Organization ID (optional)

### 3. Run Analysis
```bash
jupyter notebook scripts/TE_autism.ipynb
```

## Key Features

### Demographic Profile Generation
- Creates 100 unique profiles based on research data
- **Age distribution**: 25-50 years (average 33.4)
- **Gender**: 51% female, 49% male
- **Locations**: US (76%), Canada (8%), UK (6%), others
- **Autism experience**: Caregiver (33%), Professional (4%), Some (30%), None (30%)
- **Knowledge levels**: None (6%), Little (65%), A lot (29%)

### AI Response Simulation
- **Multi-model support**: GPT-4, GPT-4-turbo, GPT-3.5-turbo
- **Context-aware prompting**: Incorporates demographic backgrounds
- **Length control**: Based on human response averages (49.27 words)
- **Cost optimization**: Reduces API calls from 3800 to 920 (76% savings)

### Multi-Tier Rating System
- **Researchers** (n=2): Evaluate ALL responses on 5 dimensions
- **Individuals with autism** (n=6): Evaluate 50 responses each for helpfulness
- **Autism experts** (n=11): Evaluate 20 responses each for helpfulness
- **Enhanced LLM rating**: Uses calibration examples for consistent evaluation

### Statistical Analysis
- **Inter-rater reliability**: Krippendorff's α calculation
- **Group comparisons**: Mann-Whitney U tests with effect sizes
- **Automated reporting**: Generate publication-ready tables

## Evaluation Dimensions

1. **Directness** (0/1): Provides clear, actionable guidance
2. **Additional Information** (0/1): Offers context beyond the question
3. **Informational Support** (0/1): Gives practical advice
4. **Emotional Support** (0/1): Shows empathy and validation
5. **Helpfulness** (1-5): Overall utility rating

## Expected Results

### Inter-rater Reliability (Krippendorff's α)
- **Target range**: 0.6-0.8 (Good reliability)
- **Directness**: α = 0.735
- **Additional Information**: α = 0.755
- **Informational Support**: α = 0.887
- **Emotional Support**: α = 0.612

### Cost Efficiency
- **Full evaluation**: ~$5.70 (3,800 LLM calls)
- **Cost-controlled**: ~$1.38 (920 LLM calls)
- **Savings**: 76% cost reduction while maintaining statistical power

### Data Volume
- **Questions**: 20 autism community questions
- **Human responses**: 200 (100 in-group + 100 out-group)
- **AI responses**: 100-400 per model
- **Total evaluations**: 920 rating instances

## Usage Example

```python
# 1. Generate demographic profiles
responder_profiles = generate_responder_profiles(100)

# 2. Run AI simulation
results = run_multi_model_comparison(target_answers=400)

# 3. Execute rating simulation
rating_results = run_complete_llm_rating_simulation(
    model_name="gpt-4o-mini",
    individual_sample_size=50,
    expert_sample_size=20
)

# 4. Analyze results
alpha_results = calculate_krippendorff_alpha()
comparison_results = analyze_group_comparisons()
```

## Important Notes

- Requires OpenAI API access and billing setup
- API key files are gitignored for security
- Cost-controlled mode recommended for budget management
- Results include both raw data and publication-ready tables

## Dependencies

**Core packages**: pandas, numpy, scipy, openai, krippendorff
**Analysis**: statsmodels, matplotlib, seaborn
**Environment**: jupyter, tqdm, python-dotenv

## Contact

- **GitHub**: [@CWY88](https://github.com/CWY88)
- **Repository**: [Crowdsource-Turing-Experiment](https://github.com/CWY88/Crowdsource-Turing-Experiment)

---

*Advancing AI evaluation through community-centered research*
