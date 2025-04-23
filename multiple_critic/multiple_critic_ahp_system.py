import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def ahp_attributes(ahp_df):
    """
    Calculates priority indices from a pairwise comparison matrix using the AHP method.
    """
    sum_array = np.array(ahp_df.sum(numeric_only=True))
    cell_by_sum = ahp_df.div(sum_array, axis=1)
    priority_df = pd.DataFrame(cell_by_sum.mean(axis=1), index=ahp_df.index, columns=['Priority Index'])
    priority_df = priority_df.transpose()
    return priority_df

def alternative_priority_index(alternative_attr_df, attr_name):
    """
    Calculates priority indices for alternatives under a specific criterion.
    """
    data_dict = {}
    data_dict[f"ahp_df_alternative_{attr_name}"] = alternative_attr_df.loc[attr_name]
    data_dict[f"sum_array_alternative_{attr_name}"] = np.array(data_dict[
        f"ahp_df_alternative_{attr_name}"].sum(numeric_only=True))
    data_dict[f"norm_mat_alternative_{attr_name}"] = data_dict[
        f"ahp_df_alternative_{attr_name}"].div(data_dict[f"sum_array_alternative_{attr_name}"], axis=1)
    priority_df = pd.DataFrame(data_dict[
        f"norm_mat_alternative_{attr_name}"].mean(axis=1),
                               index=alternative_attr_df.loc[attr_name].index, columns=[attr_name])
    return priority_df

def plot_alternative_ranking(final_alternatives_sum_df):
    """
    Plots the final priority indices of the alternatives using a bar plot.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(y=final_alternatives_sum_df['Sum'],
                x=final_alternatives_sum_df.index,
                palette='viridis')
    plt.title('Final Alternatives Priority Index')
    plt.ylabel('Priority Index')
    plt.xlabel('Alternatives')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_criteria_weights(priority_index_criteria):
    """
    Plots the weights of the criteria using a pie chart.
    """
    plt.figure(figsize=(10, 6))
    plt.pie(priority_index_criteria.values[0],
            labels=priority_index_criteria.columns,
            autopct='%1.1f%%',
            colors=sns.color_palette('viridis'),
            startangle=90)
    plt.title('Criteria Weights Distribution')
    plt.show()

def perform_sensitivity_analysis(priority_index_criteria, final_alternatives_df):
    """
    Performs sensitivity analysis on criteria weights to assess the robustness of alternative rankings.
    """
    N = 10**4
    s_values = np.arange(0.2, 0.7, 0.1)
    
    original_weights = np.array(priority_index_criteria.loc['Priority Index'])
    original_scores = final_alternatives_df.multiply(original_weights, axis=1).sum(axis=1)
    original_ranking = original_scores.sort_values(ascending=False).index.tolist()
    
    alternative_results = {alt: [] for alt in final_alternatives_df.index}
    prr_values = []
    
    for s in s_values:
        alternative_scores = {alt: [] for alt in final_alternatives_df.index}
        rank_reversals = 0
        
        for _ in range(N):
            perturbations = np.random.uniform(-s, s, size=len(original_weights))
            perturbed_weights = original_weights + perturbations
            perturbed_weights = np.maximum(perturbed_weights, 1e-10)
            perturbed_weights = perturbed_weights / np.sum(perturbed_weights)
            
            perturbed_scores = final_alternatives_df.multiply(perturbed_weights, axis=1).sum(axis=1)
            perturbed_ranking = perturbed_scores.sort_values(ascending=False).index.tolist()
            
            if perturbed_ranking != original_ranking:
                rank_reversals += 1
            
            for alt in final_alternatives_df.index:
                alternative_scores[alt].append(perturbed_scores[alt])
        
        prr = rank_reversals / N
        prr_values.append(prr)
        
        for alt in final_alternatives_df.index:
            mean_score = np.mean(alternative_scores[alt])
            alternative_results[alt].append(mean_score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(s_values, prr_values, marker='o')
    plt.xlabel('Perturbation Strength (s)')
    plt.ylabel('Probability of Rank Reversal (PRR)')
    plt.title('Sensitivity Analysis: PRR vs Perturbation Strength')
    plt.grid(True)
    plt.show()
    
    plot_enhanced_sensitivity_analysis(s_values, alternative_results, original_scores)
    plot_rankings_comparison(alternative_results, original_scores, s_values)

def plot_enhanced_sensitivity_analysis(s_values, alternative_results, original_scores):
    """
    Plots sensitivity patterns for each alternative, including a ranking text box.
    """
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111)
    
    for alt in alternative_results.keys():
        ax.plot(s_values, alternative_results[alt], marker='o', label=f'{alt}')
        ax.scatter(0, original_scores[alt], marker='s')
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Perturbation Strength (s)')
    ax.set_ylabel('Priority Score')
    ax.set_title('Sensitivity Analysis: Alternative Priorities vs Perturbation Strength')
    ax.legend(title='Alternatives')
    ax.grid(True)
    
    initial_ranking = original_scores.sort_values(ascending=False)
    ranking_text = "Initial Rankings:\n" + "\n".join(
        [f"{i+1}. {alt}: {score:.3f}"
         for i, (alt, score) in enumerate(initial_ranking.items())])
    
    plt.figtext(0.85, 0.5, ranking_text,
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
                verticalalignment='center')
    
    plt.subplots_adjust(right=0.82)
    plt.show()

def plot_rankings_comparison(alternative_results, original_scores, s_values):
    """
    Compares initial and final alternative priorities using a bar plot.
    """
    plt.figure(figsize=(10, 6))
    final_scores = {alt: results[-1] for alt, results in alternative_results.items()}
    alternatives = list(original_scores.index)
    x = np.arange(len(alternatives))
    width = 0.35
    
    plt.bar(x - width/2, original_scores, width, label='Initial Priorities', color='green')
    plt.bar(x + width/2, final_scores.values(), width, label=f'Final Priorities (s=0.6)', color='yellow')
    
    plt.xlabel('Alternatives')
    plt.ylabel('Priority Score')
    plt.title('Comparison of Initial vs Final Alternative Priorities')
    plt.xticks(x, alternatives)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for i, v in enumerate(original_scores):
        plt.text(i - width/2, v, f'{v:.3f}', ha='center', va='bottom')
    for i, v in enumerate(final_scores.values()):
        plt.text(i + width/2, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def perform_complete_sensitivity_analysis(priority_index_criteria, final_alternatives_df, alternatives_df):
    """
    Performs sensitivity analysis for both criteria weights and pairwise comparison judgments.
    """
    print("\nScenario 1: Sensitivity Analysis - Criteria Weights\nRunning...")
    perform_sensitivity_analysis(priority_index_criteria, final_alternatives_df)
    print("\nScenario 1 Completed.\n")
    print("\nScenario 2: Sensitivity Analysis - Pairwise Comparisons\nRunning...")
    perform_pairwise_sensitivity_analysis(alternatives_df, priority_index_criteria)
    print("\nScenario 2 Completed.\n")

def perform_pairwise_sensitivity_analysis(alternatives_df, priority_index_criteria):
    """
    Performs sensitivity analysis by perturbing pairwise comparison values.
    """
    N = 10**4
    s_values = np.arange(0.2, 0.7, 0.1)
    
    original_alternatives = process_alternatives(alternatives_df, priority_index_criteria)
    original_ranking = original_alternatives.sort_values('Sum', ascending=False).index.tolist()
    
    prr_values = []
    alternative_results = {alt: [] for alt in original_alternatives.index}
    
    for s in s_values:
        rank_reversals = 0
        temp_scores = {alt: [] for alt in original_alternatives.index}
        
        for _ in range(N):
            perturbed_df = perturb_pairwise_comparisons(alternatives_df, s)
            perturbed_scores = process_alternatives(perturbed_df, priority_index_criteria)
            perturbed_ranking = perturbed_scores.sort_values('Sum', ascending=False).index.tolist()
            
            if perturbed_ranking != original_ranking:
                rank_reversals += 1
            
            for alt in perturbed_scores.index:
                temp_scores[alt].append(perturbed_scores.loc[alt, 'Sum'])
        
        prr = rank_reversals / N
        prr_values.append(prr)
        
        for alt in alternative_results:
            alternative_results[alt].append(np.mean(temp_scores[alt]))
    
    plot_pairwise_sensitivity_results(s_values, prr_values, alternative_results, original_alternatives)

def perturb_pairwise_comparisons(alternatives_df, s):
    """
    Creates a perturbed version of the pairwise comparison matrices.
    """
    perturbed_df = alternatives_df.copy()
    
    for criteria in ['Effectiveness', 'Compliance', 'Implementation', 'Cost']:
        matrix = perturbed_df.loc[criteria]
        
        for i in matrix.index:
            for j in matrix.columns:
                if i != j:
                    perturbation = np.random.uniform(-s, s)
                    original_value = matrix.loc[i, j]
                    new_value = original_value * (1 + perturbation)
                    new_value = max(new_value, 0.1)
                    perturbed_df.loc[(criteria, i), j] = new_value
    
    return perturbed_df

def process_alternatives(alternatives_df, priority_index_criteria):
    """
    Process alternatives data to get final scores.
    """
    alternative_dfs = []
    for criteria in ['Effectiveness', 'Compliance', 'Implementation', 'Cost']:
        alt_df = alternative_priority_index(alternatives_df, criteria)
        alternative_dfs.append(alt_df)
    
    final_alt_df = pd.concat(alternative_dfs, axis=1)
    final_sum_df = final_alt_df.multiply(np.array(priority_index_criteria.loc['Priority Index']), axis=1)
    final_sum_df['Sum'] = final_sum_df.sum(axis=1)
    return final_sum_df

def plot_pairwise_sensitivity_results(s_values, prr_values, alternative_results, original_scores):
    """
    Plots the results of pairwise comparison sensitivity analysis.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(s_values, prr_values, marker='o')
    plt.xlabel('Perturbation Strength (s)')
    plt.ylabel('Probability of Rank Reversal (PRR)')
    plt.title('Sensitivity Analysis: PRR vs Perturbation Strength (Pairwise Comparisons)')
    plt.grid(True)
    plt.show()
    
    plot_enhanced_sensitivity_analysis(s_values, alternative_results, original_scores['Sum'])

def consistency_ratio(priority_index, ahp_df, logging=False):
    """
    Calculates the consistency ratio to check the consistency of the pairwise comparison matrix.
    """
    random_matrix = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32,
                     8: 1.14, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56,
                     14: 1.57, 15: 1.59, 16: 1.605, 17: 1.61, 18: 1.615, 19: 1.62, 20: 1.625}
    
    consistency_df = ahp_df.multiply(np.array(priority_index.loc['Priority Index']), axis=1)
    consistency_df['sum_of_col'] = consistency_df.sum(axis=1)
    lambda_max_df = consistency_df['sum_of_col'].div(np.array(priority_index.transpose()
                                                              ['Priority Index']), axis=0)
    lambda_max = lambda_max_df.mean()
    consistency_index = round((lambda_max - len(ahp_df.index)) / (len(ahp_df.index) - 1), 3)
    consistency_ratio = round(consistency_index / random_matrix[len(ahp_df.index)], 3)
    if logging:
        print(f'    Consistency Ratio: {consistency_ratio}')
        print(f'    Consistency Index: {consistency_index}')
    return consistency_ratio < 0.1

def check_consistency_ratio(logging=False):
    """
    Check the consistency ratio of both criteria and alternatives matrices for every critic.
    """
    critics_consistent = True
    for i in range(1, 11):
        criteria_file = f'multiple_critic/csv_files/critic{i}/criteria_comparison_decimal.csv'
        criteria_df = pd.read_csv(criteria_file, index_col=0)
        if logging:
            print(f"\nCritic {i}:")
            print("\n    Criteria Matrix:")
        priority_index_criteria = ahp_attributes(criteria_df)
        if not consistency_ratio(priority_index_criteria, criteria_df, logging):
            print(f'Critic {i} criteria matrix is not consistent')
            critics_consistent = False
        
        alternatives_file = f'multiple_critic/csv_files/critic{i}/alternatives_comparison_decimal.csv'
        alternatives_df = pd.read_csv(alternatives_file, index_col=[0, 1])
        
        for criterion in ['Effectiveness', 'Compliance', 'Implementation', 'Cost']:
            if logging:
                print(f"\n    {criterion} Alternatives Matrix:")
            alt_df = alternatives_df.loc[criterion]
            alt_priority = ahp_attributes(alt_df)
            if not consistency_ratio(alt_priority, alt_df, logging):
                print(f'Critic {i} alternatives matrix for {criterion} is not consistent')
                critics_consistent = False
    
    if critics_consistent:
        print("\nAll critics have consistent matrices. Proceeding with analysis...\n")
    else:
        print(f'\nERROR! Not all critics have consistent matrices. Terminating...\n')
        exit(1)

def aggregate_priorities(priority_list):
    """Aggregates priorities using geometric mean"""
    priorities_array = np.array(priority_list)
    geometric_mean = np.exp(np.mean(np.log(priorities_array), axis=0))
    return geometric_mean / np.sum(geometric_mean)

def main():
    """
    Main function to execute the AHP process.
    """
    check_consistency_ratio(logging=False)
    
    all_criteria_priorities = []
    all_alternative_priorities = {
        'Effectiveness': [],
        'Compliance': [],
        'Implementation': [],
        'Cost': []
    }
    
    for i in range(1, 11):
        criteria_file = f'multiple_critic/csv_files/critic{i}/criteria_comparison_decimal.csv'
        criteria_df = pd.read_csv(criteria_file, index_col=0)
        priority_index_criteria = ahp_attributes(criteria_df)
        all_criteria_priorities.append(priority_index_criteria.loc['Priority Index'])
        
        alternatives_file = f'multiple_critic/csv_files/critic{i}/alternatives_comparison_decimal.csv'
        alternatives_df = pd.read_csv(alternatives_file, index_col=[0, 1])
        
        for criterion in ['Effectiveness', 'Compliance', 'Implementation', 'Cost']:
            alt_df = alternative_priority_index(alternatives_df, criterion)
            all_alternative_priorities[criterion].append(alt_df[criterion])
    
    aggregated_criteria = aggregate_priorities(all_criteria_priorities)
    aggregated_criteria_df = pd.DataFrame([aggregated_criteria],
                                           columns=['Effectiveness', 'Compliance', 'Implementation', 'Cost'],
                                           index=['Priority Index'])
    print("\nAggregated Criteria Priorities:")
    print(aggregated_criteria_df)
    
    aggregated_alternatives = {}
    for criterion in ['Effectiveness', 'Compliance', 'Implementation', 'Cost']:
        aggregated_alternatives[criterion] = aggregate_priorities(all_alternative_priorities[criterion])
    
    final_alternatives_df = pd.DataFrame(aggregated_alternatives,
                                          index=['Multi-factor Authentication', 'Network Security', 'Data Encryption'])
    
    final_alternatives_sum_df = final_alternatives_df.multiply(
        np.array(aggregated_criteria_df.loc['Priority Index']), axis=1)
    final_alternatives_sum_df['Sum'] = final_alternatives_sum_df.sum(axis=1)
    final_alternatives_sum_df = final_alternatives_sum_df.sort_values('Sum', ascending=False)
    
    print("\nFinal Aggregated Results:")
    print(final_alternatives_sum_df)
    
    plot_criteria_weights(aggregated_criteria_df)
    plot_alternative_ranking(final_alternatives_sum_df)
    
    perform_complete_sensitivity_analysis(aggregated_criteria_df,
                                           final_alternatives_df,
                                           alternatives_df)

if __name__ == "__main__":
    main()