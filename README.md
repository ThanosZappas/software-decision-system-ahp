# Multiple Critic AHP System

This project implements a **Multiple Critic Analytical Hierarchy Process (AHP)** system for evaluating and ranking cybersecurity software alternatives based on multiple criteria. The system aggregates judgments from multiple critics to provide a comprehensive decision-making framework.

## Features

- **Multi-Critic Support**: Combines judgments from multiple critics to ensure a balanced evaluation.
- **Criteria and Alternatives Comparison**: Uses pairwise comparison matrices for criteria and alternatives.
- **Consistency Check**: Validates the consistency of judgments using the Consistency Ratio.
- **Priority Aggregation**: Aggregates priorities using the geometric mean for criteria and alternatives.
- **Sensitivity Analysis**: Performs sensitivity analysis to evaluate the robustness of rankings under perturbations.
- **Visualization**: Provides bar plots, pie charts, and sensitivity analysis graphs for better insights.

## Criteria

The system evaluates alternatives based on the following criteria:

1. **Effectiveness**: Measures the reduction in cybersecurity incidents.
2. **Compliance**: Assesses compatibility with government regulations.
3. **Ease of Implementation**: Evaluates the ease of integration and deployment.
4. **Cost**: Considers the total implementation and maintenance cost.

## Alternatives

The alternatives being evaluated are:

1. **Multi-factor Authentication**
2. **Network Security**
3. **Data Encryption**

## How It Works

1. **Input**: Critics provide pairwise comparison matrices for criteria and alternatives.
2. **Consistency Check**: Ensures that the input matrices are consistent.
3. **Priority Calculation**: Calculates priority indices for criteria and alternatives.
4. **Aggregation**: Aggregates priorities from all critics using the geometric mean.
5. **Final Ranking**: Computes the final ranking of alternatives based on aggregated priorities.
6. **Sensitivity Analysis**: Tests the robustness of rankings by perturbing criteria weights and pairwise comparisons.

## Usage

1. Place the pairwise comparison matrices for each critic in the `csv_files` folder under their respective subfolders (e.g., `critic1`, `critic2`, etc.).
2. Run the `multiple_critic_ahp_system.py` script inside a conda environment:
   ```bash
   python multiple_critic_ahp_system.py
   ```

## Results

### Aggregated Criteria Priorities
| Criterion              | Weight  |
|------------------------|---------|
| Effectiveness          | 0.405   |
| Compliance             | 0.166   |
| Implementation         | 0.138   |
| Cost                   | 0.291   |

### Final Aggregated Results
| Alternative                  | Effectiveness | Compliance | Implementation | Cost   | Sum   |
|------------------------------|---------------|------------|----------------|--------|-------|
| Multi-factor Authentication  | 0.139         | 0.060      | 0.063          | 0.108  | 0.370 |
| Network Security             | 0.146         | 0.059      | 0.029          | 0.102  | 0.336 |
| Data Encryption              | 0.120         | 0.047      | 0.046          | 0.081  | 0.293 |

### Final Alternative Rankings
| Alternative                  | Priority Score |
|------------------------------|----------------|
| Multi-factor Authentication  | 0.370          |
| Network Security             | 0.336          |
| Data Encryption              | 0.293          |

### Sensitivity Analysis

The system performs two types of sensitivity analysis:
1. **Criteria Weight Perturbation**: Evaluates the impact of changes in criteria weights on the rankings.
2. **Pairwise Comparison Perturbation**: Assesses the effect of changes in pairwise comparison judgments on the rankings.

## Outputs

- **Aggregated Criteria Priorities**: Displays the combined weights for each criterion.
- **Final Alternative Rankings**: Shows the overall ranking of alternatives.
- **Visualization**: Generates plots for criteria weights, alternative rankings, and sensitivity analysis.

## Dependencies

- Python 3.11.7 was used for this project (It is possible that it's compatible on different versions aswell.)
- Required libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`

Install dependencies using:
```bash
pip install pandas numpy matplotlib seaborn
```

## Author

Developed by Georgios Ivantsos and Zappas Athanasios.