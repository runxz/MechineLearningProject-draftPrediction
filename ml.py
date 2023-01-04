import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the data
data = pd.read_csv("match_data.csv")

# Split the data into features and labels
X = data[['team_1_player_1', 'team_1_player_2', 'team_1_player_3', 'team_1_player_4','team_1_player_5','team_2_player_1', 'team_2_player_2', 'team_2_player_3','team_2_player_4','team_2_player_5']]
y = data['result']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a random forest classifier
clf = RandomForestClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Get user input for the team compositions
team_1 = input("Enter the players for team 1 separated by commas: ")
team_2 = input("Enter the players for team 2 separated by commas: ")

# Split the team compositions into lists of players
team_1_players = team_1.split(",")
team_2_players = team_2.split(",")

# Create a feature array for the user input
X_input = [team_1_players + team_2_players]

# Make a prediction using the classifier
prediction = clf.predict(X_input)

# Print the prediction
if prediction == "win":
    print("The model predicts that team 1 will win the match.")
else:
    print("The model predicts that team 2 will win the match.")
