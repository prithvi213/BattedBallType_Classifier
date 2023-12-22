import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
import seaborn as sns
from sklearn.metrics import accuracy_score

def main():
    # Load data into dataframes and clean each dataset
    flyballs_df_desc = pd.read_csv('../datafiles/flyballs.csv')
    flyballs_df_asc = pd.read_csv('../datafiles/flyballs_lowest.csv')
    flyballs_df = pd.concat([flyballs_df_desc, flyballs_df_asc])
    flyballs_df = flyballs_df.drop_duplicates()

    grounders_df_desc = pd.read_csv('../datafiles/grounders.csv')
    grounders_df_asc = pd.read_csv('../datafiles/grounders_lowest.csv')
    grounders_df_87_90 = pd.read_csv('../datafiles/grounders_87_90.csv')
    grounders_df = pd.concat([grounders_df_desc, grounders_df_asc, grounders_df_87_90])
    grounders_df = grounders_df.drop_duplicates()

    liners_df_desc = pd.read_csv('../datafiles/liners.csv')
    liners_df_asc = pd.read_csv('../datafiles/liners_lowest.csv')
    liners_df = pd.concat([liners_df_desc, liners_df_asc])
    liners_df = liners_df.drop_duplicates()

    popups_df = pd.read_csv('../datafiles/popups.csv')
    smallest_df_size = popups_df.shape[0]

    # Reshaped 3 of the datasets due to popups dataframe having significantly smaller size
    flyballs_df_sample = flyballs_df.sample(smallest_df_size, random_state=10)
    grounders_df_sample = grounders_df.sample(smallest_df_size, random_state=10)
    liners_df_sample = liners_df.sample(smallest_df_size, random_state=10)
    
    flyballs_exit_velo = np.array(flyballs_df_sample['launch_speed'].values)
    flyballs_la = np.array(flyballs_df_sample['launch_angle'].values)
    flyballs_type = np.array(flyballs_df_sample['bb_type'].values)

    grounders_exit_velo = np.array(grounders_df_sample['launch_speed'].values)
    grounders_la = np.array(grounders_df_sample['launch_angle'].values)
    grounders_type = np.array(grounders_df_sample['bb_type'].values)

    liners_exit_velo = np.array(liners_df_sample['launch_speed'].values)
    liners_la = np.array(liners_df_sample['launch_angle'].values)
    liners_type = np.array(liners_df_sample['bb_type'].values)

    popups_exit_velo = np.array(popups_df['launch_speed'].values)
    popups_la = np.array(popups_df['launch_angle'].values)
    popups_type = np.array(popups_df['bb_type'].values)
    
    exit_velos = np.concatenate((flyballs_exit_velo, grounders_exit_velo, liners_exit_velo, popups_exit_velo))
    launch_angles = np.concatenate((flyballs_la, grounders_la, liners_la, popups_la))

    X = np.column_stack((exit_velos, launch_angles))
    y = np.concatenate((flyballs_type, grounders_type, liners_type, popups_type))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=50)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    color_map = {"fly_ball": "orange",
                 "ground_ball": "green",
                 "line_drive": "yellow",
                 "popup": "blue"}
    
    blue_xvalues = [X_train[i][0] for i in range(len(X_train)) if color_map[y_train[i]] == "blue"]
    blue_yvalues = [X_train[i][1] for i in range(len(X_train)) if color_map[y_train[i]] == "blue"]
    orange_xvalues = [X_train[i][0] for i in range(len(X_train)) if color_map[y_train[i]] == "orange"]
    orange_yvalues = [X_train[i][1] for i in range(len(X_train)) if color_map[y_train[i]] == "orange"]
    green_xvalues = [X_train[i][0] for i in range(len(X_train)) if color_map[y_train[i]] == "green"]
    green_yvalues = [X_train[i][1] for i in range(len(X_train)) if color_map[y_train[i]] == "green"]
    yellow_xvalues = [X_train[i][0] for i in range(len(X_train)) if color_map[y_train[i]] == "yellow"]
    yellow_yvalues = [X_train[i][1] for i in range(len(X_train)) if color_map[y_train[i]] == "yellow"]

    plt.grid(True)
    #plt.scatter(x_values, y_values, c=color_array, s=50, edgecolors='k', linewidth=0.5)
    plt.title("Batted Ball Types vs. Exit Velocity and Launch Angle")
    plt.xlabel("Exit Velocity (mph)")
    plt.ylabel("Launch Angle (deg)")
    sns.kdeplot(x=blue_xvalues, y=blue_yvalues, cmap='Blues', fill=True, alpha=0.9)
    sns.kdeplot(x=orange_xvalues, y=orange_yvalues, cmap='Oranges', fill=True, alpha=0.9)
    sns.kdeplot(x=green_xvalues, y=green_yvalues, cmap='Greens', fill=True, alpha=0.9)
    sns.kdeplot(x=yellow_xvalues, y=yellow_yvalues, cmap='YlOrBr', fill=True, alpha=0.9)
    
    y_pred = clf.predict(X_test)
    x_values = [X_test[i][0] for i in range(len(X_test))]
    y_values = [X_test[i][1] for i in range(len(X_test))]
    color_array = [color_map[y_pred[i]] for i in range(len(y_pred))]

    accuracy = (accuracy_score(y_test, y_pred) * 100)
    print(f"Accuracy: {accuracy:.1f}%") 

    plt.scatter(x_values, y_values, c=color_array, s=2, edgecolors='k', linewidth=0.25)
    plt.show()

    

if __name__ == "__main__":
    main()