# Importing necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# def convert_fraction(val):
#     if isinstance(val, str) and '/' in val:
#         num, denom = val.split('/')
#         return float(num) / float(denom)
#     return float(val)

# We will introduce a function to find the priority index. 
# Then we provide the attributes data to this function.
def ahp_attributes(ahp_df):
    # Creating an array of sum of values in each column
    sum_array = np.array(ahp_df.sum(numeric_only=True))
    # Creating a normalized pairwise comparison matrix.
    # By dividing each column cell value with the sum of the respective column.
    cell_by_sum = ahp_df.div(sum_array,axis=1)
    # Creating Priority index by taking avg of each row
    priority_df = pd.DataFrame(cell_by_sum.mean(axis=1),
                               index=ahp_df.index,columns=['Priority Index'])
    priority_df = priority_df.transpose()
    return priority_df

def consistency_ratio(priority_index,ahp_df):
    random_matrix = {1:0,2:0,3:0.58,4:0.9,5:1.12,6:1.24,7:1.32,
                     8:1.14,9:1.45,10:1.49,11:1.51,12:1.48,13:1.56,
                     14:1.57,15:1.59,16:1.605,17:1.61,18:1.615,19:1.62,20:1.625}
    # Check for consistency
    consistency_df = ahp_df.multiply(np.array(priority_index.loc['Priority Index']),axis=1)
    consistency_df['sum_of_col'] = consistency_df.sum(axis=1)
    # To find lambda max
    lambda_max_df = consistency_df['sum_of_col'].div(np.array(priority_index.transpose()
                                                              ['Priority Index']),axis=0)
    lambda_max = lambda_max_df.mean()
    # To find the consistency index
    consistency_index = round((lambda_max-len(ahp_df.index))/(len(ahp_df.index)-1),3)
    print(f'The Consistency Index is: {consistency_index}')
    # To find the consistency ratio
    consistency_ratio = round(consistency_index/random_matrix[len(ahp_df.index)],3)
    print(f'The Consistency Ratio is: {consistency_ratio}')
    if consistency_ratio<0.1:
        print('The model is consistent')
    else:
        print('The model is not consistent')
        
def alternative_priority_index(alternative_attr_df,attr_name):
    data_dict = {}
    # To find supplier priority indices
    # Supplier priority for attr 1
    data_dict[f"ahp_df_alternative_{attr_name}"] = alternative_attr_df.loc[attr_name]
    # Creating an array of sum of values in each column
    data_dict[f"sum_array_alternative_{attr_name}"] = np.array(data_dict[
        f"ahp_df_alternative_{attr_name}"].sum(numeric_only=True))
    # Normalised pairwise comparison matrix
    # Dividing each column cell value with the sum of the respective column.
    data_dict[f"norm_mat_alternative_{attr_name}"] = data_dict[
        f"ahp_df_alternative_{attr_name}"].div(data_dict[f"sum_array_alternative_{attr_name}"],axis=1)
    priority_df = pd.DataFrame(data_dict[
        f"norm_mat_alternative_{attr_name}"].mean(axis=1),
                               index=alternative_attr_df.loc[attr_name].index,columns=[attr_name])
    return priority_df

def plot_alternative_ranking(final_alternatives_sum_df):
        # Plotting the final alternatives
        plt.figure(figsize=(10, 6))
        sns.barplot(y=final_alternatives_sum_df['Sum'], x=final_alternatives_sum_df.index, palette='viridis')
        plt.title('Final Alternatives Priority Index')
        plt.ylabel('Priority Index')
        plt.xlabel('Alternatives')
        plt.grid(axis='y')
        plt.show()

def plot_criteria_weights(priority_index_criteria):
    plt.figure(figsize=(10, 6))
    # Create pie chart for criteria weights
    plt.pie(priority_index_criteria.values[0], 
            labels=priority_index_criteria.columns,
            autopct='%1.1f%%',
            colors=sns.color_palette('viridis'), 
            startangle=90)
    plt.title('Criteria Weights Distribution')
    plt.show()

def perform_sensitivity_analysis(priority_index_criteria, final_alternatives_df):
    """
    Performs sensitivity analysis on criteria weights and compares alternatives with PRR
    """
    N = 10**4
    s_values = np.arange(0.2, 0.7, 0.1)
    
    # Get original ranking and scores
    original_weights = np.array(priority_index_criteria.loc['Priority Index'])
    original_scores = final_alternatives_df.multiply(original_weights, axis=1).sum(axis=1)
    original_ranking = original_scores.sort_values('Sum', ascending=False).index.tolist()
    
    # Store results and PRR for each perturbation strength
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
            perturbed_ranking = perturbed_scores.sort_values('Sum', ascending=False).index.tolist()
            
            # Simpler rank reversal check
            if perturbed_ranking != original_ranking:
                rank_reversals += 1
            
            # Store scores for each alternative
            for alt in final_alternatives_df.index:
                alternative_scores[alt].append(perturbed_scores[alt])
        
        # Calculate PRR for this perturbation strength
        prr = rank_reversals / N
        prr_values.append(prr)
        
        # Calculate mean score for each alternative
        for alt in final_alternatives_df.index:
            mean_score = np.mean(alternative_scores[alt])
            alternative_results[alt].append(mean_score)
    
    # Plot PRR
    plt.figure(figsize=(10, 6))
    plt.plot(s_values, prr_values, marker='o')
    plt.xlabel('Perturbation Strength (s)')
    plt.ylabel('Probability of Rank Reversal (PRR)')
    plt.title('Sensitivity Analysis: PRR vs Perturbation Strength')
    plt.grid(True)
    plt.show()
    
    # Create original plots
    plot_enhanced_sensitivity_analysis(s_values, alternative_results, original_scores)
    plot_rankings_comparison(alternative_results, original_scores, s_values)

def plot_enhanced_sensitivity_analysis(s_values, alternative_results, original_scores):
    """
    Enhanced plotting showing all alternatives' sensitivity patterns with rankings text box
    """
    fig = plt.figure(figsize=(15, 6))
    
    # Create main plot
    ax = fig.add_subplot(111)
    
    # Plot sensitivity lines for each alternative
    for alt in alternative_results.keys():
        ax.plot(s_values, alternative_results[alt], marker='o', label=f'{alt}')
        # Plot initial score as a point on y-axis
        ax.scatter(0, original_scores[alt], marker='s')
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Perturbation Strength (s)')
    ax.set_ylabel('Priority Score')
    ax.set_title('Sensitivity Analysis: Alternative Priorities vs Perturbation Strength')
    ax.legend(title='Alternatives')
    ax.grid(True)
    
    # Add text box with rankings
    initial_ranking = original_scores.sort_values(ascending=False)
    ranking_text = "Initial Rankings:\n" + "\n".join(
        [f"{i+1}. {alt}: {score:.3f}" 
         for i, (alt, score) in enumerate(initial_ranking.items())])
    
    # Position the text box outside the plot
    plt.figtext(0.85, 0.5, ranking_text,
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
                verticalalignment='center')
    
    # Adjust layout to prevent text box overlap
    plt.subplots_adjust(right=0.82)
    plt.show()

def plot_rankings_comparison(alternative_results, original_scores, s_values):
    """
    Create a bar plot comparing initial and final rankings
    """
    plt.figure(figsize=(10, 6))
    
    # Get final scores (at maximum perturbation)
    final_scores = {alt: results[-1] for alt, results in alternative_results.items()}
    
    # Prepare data for plotting
    alternatives = list(original_scores.index)
    x = np.arange(len(alternatives))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, original_scores, width, label='Initial Priorities', color='green')
    plt.bar(x + width/2, final_scores.values(), width, label=f'Final Priorities (s=0.6)', color='yellow')
    
    # Customize plot
    plt.xlabel('Alternatives')
    plt.ylabel('Priority Score')
    plt.title('Comparison of Initial vs Final Alternative Priorities')
    plt.xticks(x, alternatives)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(original_scores):
        plt.text(i - width/2, v, f'{v:.3f}', ha='center', va='bottom')
    for i, v in enumerate(final_scores.values()):
        plt.text(i + width/2, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def perform_complete_sensitivity_analysis(priority_index_criteria, final_alternatives_df, alternatives_df):
    """
    Performs sensitivity analysis for both scenarios:
    1. Changing criteria weights
    2. Changing pairwise comparison judgments
    """
    # Scenario 1: Weight perturbation (existing analysis)
    print("\nScenario 1: Sensitivity Analysis - Criteria Weights")
    perform_sensitivity_analysis(priority_index_criteria, final_alternatives_df)
    
    # Scenario 2: Pairwise comparison perturbation
    print("\nScenario 2: Sensitivity Analysis - Pairwise Comparisons")
    perform_pairwise_sensitivity_analysis(alternatives_df, priority_index_criteria)

def perform_pairwise_sensitivity_analysis(alternatives_df, priority_index_criteria):
    """
    Performs sensitivity analysis by perturbing pairwise comparison values
    """
    N = 10**4
    s_values = np.arange(0.2, 0.7, 0.1)
    
    # Get original scores
    original_alternatives = process_alternatives(alternatives_df, priority_index_criteria)
    original_ranking = original_alternatives.sort_values('Sum', ascending=False).index.tolist()
    
    prr_values = []
    alternative_results = {alt: [] for alt in original_alternatives.index}
    
    for s in s_values:
        rank_reversals = 0
        temp_scores = {alt: [] for alt in original_alternatives.index}
        
        for _ in range(N):
            # Create perturbed version of the comparison matrices
            perturbed_df = perturb_pairwise_comparisons(alternatives_df, s)
            
            # Process perturbed matrices
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
    
    # Plot results
    plot_pairwise_sensitivity_results(s_values, prr_values, alternative_results, original_alternatives)

def perturb_pairwise_comparisons(alternatives_df, s):
    """
    Creates a perturbed version of the pairwise comparison matrices
    """
    perturbed_df = alternatives_df.copy()
    
    for criteria in ['Effectiveness', 'Compliance', 'Implementation', 'Cost']:
        matrix = perturbed_df.loc[criteria]
        
        # Apply perturbations while maintaining reciprocal property
        for i in matrix.index:
            for j in matrix.columns:
                if i != j:
                    perturbation = np.random.uniform(-s, s)
                    original_value = matrix.loc[i, j]
                    new_value = original_value * (1 + perturbation)
                    
                    # Ensure value stays positive
                    new_value = max(new_value, 0.1)
                    
                    # Update value and its reciprocal
                    perturbed_df.loc[(criteria, i), j] = new_value
    
    return perturbed_df

def process_alternatives(alternatives_df, priority_index_criteria):
    """
    Process alternatives data to get final scores
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
    Plots the results of pairwise comparison sensitivity analysis
    """
    # Plot PRR
    plt.figure(figsize=(10, 6))
    plt.plot(s_values, prr_values, marker='o')
    plt.xlabel('Perturbation Strength (s)')
    plt.ylabel('Probability of Rank Reversal (PRR)')
    plt.title('Sensitivity Analysis: PRR vs Perturbation Strength (Pairwise Comparisons)')
    plt.grid(True)
    plt.show()
    
    # Plot sensitivity patterns
    plot_enhanced_sensitivity_analysis(s_values, alternative_results, original_scores['Sum'])

def main():
    # Reading the file with the first column as index
    criteria_df = pd.read_csv('csv_files/criteria_comparison_decimal.csv', index_col=0)
    print(criteria_df, '\n')  

    # Calling the ahp_attributes function, 
    # To return a table with the priority index for each attribute.
    priority_index_criteria = ahp_attributes(criteria_df)
    print(priority_index_criteria, '\n')

    consistency_ratio(priority_index_criteria,criteria_df)

    alternatives_df = pd.read_csv('csv_files/alternatives_comparison_decimal.csv', index_col=[0,1])
    print(alternatives_df, '\n')
    
    alternative_effectiveness_df = alternative_priority_index(alternatives_df,'Effectiveness')
    alternative_compliance_df = alternative_priority_index(alternatives_df,'Compliance')
    alternative_implementation_df = alternative_priority_index(alternatives_df,'Implementation')
    alternative_cost_df = alternative_priority_index(alternatives_df,'Cost')

    final_alternatives_df = pd.concat([alternative_effectiveness_df,alternative_compliance_df,alternative_implementation_df,alternative_cost_df],axis=1)
    final_alternatives_sum_df = final_alternatives_df.multiply(np.array(priority_index_criteria.loc['Priority Index']),axis=1)
    final_alternatives_sum_df['Sum'] = final_alternatives_sum_df.sum(axis=1)
    print(final_alternatives_sum_df)

    # Visualize the results
    # plot_criteria_weights(priority_index_criteria)
    # plot_alternative_ranking(final_alternatives_sum_df)
    
    # Add complete sensitivity analysis
    perform_complete_sensitivity_analysis(priority_index_criteria, final_alternatives_df, alternatives_df)
    
main()