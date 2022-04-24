from operator import mod
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap

EVALUATION_TYPES = [
    'test_performance',
    'explainability'
]


def run_model_evaluation(data_path=None, output_path='results',use_dummy=None, evaluation_type="explainability", model_type=None, apply_smote=True):

    assert(evaluation_type in EVALUATION_TYPES)
    if apply_smote:
        smote_applied = 'With SMOTE'
        smote_file = 'smote'
    else:
        smote_applied = 'Without SMOTE'
        smote_file='no_smote'
    evaluation = None
    if use_dummy:
        file_path = os.path.join(data_path,'dummy_data','model')
    else:
        file_path = os.path.join(data_path,'real_data','model')
    # Load Saved Model & Test Data
    loaded_model = pickle.load(open(os.path.join(file_path,'trained_model.pkl'), 'rb'))
    X_test = pd.read_csv(os.path.join(file_path,'X_test.csv'))
    y_test = pd.read_csv(os.path.join(file_path,'y_test.csv'))
    y_test.columns = ['y_actual']
    y_pred = pd.DataFrame(loaded_model.predict(np.array(X_test)))
    y_pred.columns = ['y_pred']
    # Sample Test Data
    test_df = pd.concat([X_test,y_test,y_pred], axis=1)
    if test_df.shape[0]>10000:
        sampled_test_df = test_df.sample(n=10000).copy().reset_index(drop=True)
    else:
        sampled_test_df=test_df
    y_test_sample = sampled_test_df['y_actual']
    y_pred_sample = sampled_test_df['y_pred'] 
    X_test_sample = sampled_test_df.drop(['y_actual','y_pred'], axis=1)
    # Generate Classification Report
    clf_report = generate_classification_report(y_pred, y_test)
    generate_classification_heatmap(clf_report, output_path, model_type=model_type, smote_applied=smote_applied, smote_file=smote_file)
    # Generate Global Shap Results
    shap_values = generate_shap_values(loaded_model=loaded_model.predict, X_test=X_test_sample)
    generate_shap_plot(shap_values=shap_values, output_path=output_path,model_type=model_type)
    # Generate Local Shap Explanability Results
    for i in range(0,5):
        y_test_value = str(np.where(int(y_test_sample[i])==1, 'Bad Loan', 'Not a Bad Loan'))
        y_pred_value = str(np.where(int(y_pred_sample[i])==1, 'Bad Loan', 'Not a Bad Loan'))
        local_shap_plot(shap_values[i], i, output_path, y_test_value,y_pred_value,model_type=model_type)
        local_force_shap_plot(shap_values[i], i, output_path, y_test_value, y_pred_value, model_type=model_type)
    return y_test_sample, y_pred

def generate_classification_report(y_pred, y_test):
    from sklearn.metrics import classification_report
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    return clf_report

def generate_classification_heatmap(clf_report, output_path, model_type, smote_applied, smote_file):
    gen_image = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True).set_title(f'Classification Report For {model_type} {smote_applied} Model - Bad Loan Classifier')
    save_image(gen_image, file_name='classification_report.png', output_path=output_path, model_type=model_type, smote_file=smote_file)
    plt.close()
    return 'Success'

def generate_shap_values(loaded_model=None, X_test=None):
    explainer = shap.Explainer(loaded_model,X_test)
    shap_values = explainer(X_test)
    return shap_values

def generate_shap_plot(shap_values, output_path, model_type):
    shap.plots.bar(shap_values, show=False)
    plt.title(f'The Global Feature Importance For {model_type} Model')
    plt.savefig(os.path.join(output_path, 'output', 'global_bar_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    #save_image(bar_plot, file_name='bar_plot.png', output_path=output_path)
    shap.plots.beeswarm(shap_values, show=False)
    plt.title(f'The Global Feature Importance For {model_type} Model Output')
    plt.savefig(os.path.join(output_path, 'output', 'global_bee_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    shap.summary_plot(shap_values,plot_type='violin', show=False)
    plt.title(f'The Global Feature Importance For {model_type} Model')
    plt.savefig(os.path.join(output_path, 'output', 'global_summary_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    #save_image(bee_plot, file_name='bee_plot.png', output_path=output_path)
    return 'Success'
1
def local_shap_plot(shap_values, id, output_path, y_actual, y_pred, model_type):
    file_name = f'{model_type}_Local_Explanability_Listing_Id_{id}.png'
    shap.plots.waterfall(shap_values, show=False)
    plt.title(f'{model_type} Model Predicted Listing "{y_pred}". It is actually "{y_actual}"')
    plt.savefig(os.path.join(output_path, 'output', file_name), dpi=150, bbox_inches='tight')
    plt.close()
    return 'Success'

def local_force_shap_plot(shap_values, id, output_path, y_actual, y_pred, model_type):
    file_name = f'{model_type}_Bar_Plot_Local_Explanability_Listing_Id_{id}.png'
    shap.plots.bar(shap_values, show=False)
    plt.title(f'{model_type} Model Predicted Listing "{y_pred}". It is actually "{y_actual}"')
    plt.savefig(os.path.join(output_path, 'output', file_name), dpi=150, bbox_inches='tight')
    plt.close()
    return 'Success'

def save_image(gen_image, file_name='classification_report.png', output_path='results', model_type=None, smote_file=None):
    file_name = f'{model_type}_{smote_file}_{file_name}'
    plt.savefig(os.path.join(output_path, 'output', file_name), dpi=150, bbox_inches='tight')
    plt.close()
    return 'Success'