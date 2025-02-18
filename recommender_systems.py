import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class RecommenderSystem:
    def __init__(self, num_items=50, num_features=10):
        self.num_items = num_items
        self.num_features = num_features
        
        # Sample real-world music genres/features
        self.features_names = ["Acousticness", "Danceability", "Energy", "Instrumentalness", "Liveness", "Loudness", "Speechiness", "Valence", "Tempo", "Popularity"]
        
        # Simulated dataset with random values for real music features
        self.items = np.random.rand(num_items, num_features)
        self.music_names = [f"Song {i+1}" for i in range(num_items)]
        
        # Initialize user preference vector
        self.user_preferences = np.random.rand(num_features)
        
        # Normalize preferences
        self.user_preferences /= np.linalg.norm(self.user_preferences)
        
        self.history = []
        self.recommended_music = []

    def recommend(self):
        """ Recommend an item based on user preferences """
        scores = self.items @ self.user_preferences  # Dot product to get similarity scores
        best_index = np.argmax(scores)  # Recommend the item with the highest score
        self.recommended_music.append(self.music_names[best_index])
        return best_index

    def update_preferences(self, item_index, feedback):
        """ Update user preferences based on feedback (-1, 0, 1) """
        self.user_preferences += feedback * self.items[item_index] * 0.1  # Small update
        self.user_preferences /= np.linalg.norm(self.user_preferences)  # Normalize
        self.history.append(self.user_preferences.copy())

    def simulate(self, iterations=50):
        """ Simulate recommendations over time """
        for _ in range(iterations):
            rec = self.recommend()
            feedback = np.random.choice([-1, 0, 1], p=[0.05, 0.2, 0.75])  # Mostly positive feedback
            self.update_preferences(rec, feedback)

    def plot_preferences(self):
        """ Plot the evolution of user preferences over time """
        self.history = np.array(self.history)
        plt.figure(figsize=(12, 7))
        sns.set(style="darkgrid")
        
        for i, feature in enumerate(self.features_names):
            plt.plot(self.history[:, i], label=feature, linewidth=2)
        
        plt.xlabel("Time (iterations)", fontsize=14)
        plt.ylabel("Preference Value", fontsize=14)
        plt.title("Evolution of User Preferences in a Music Recommender System", fontsize=16)
        plt.legend()
        plt.show()

    def plot_recommendation_trend(self):
        """ Plot the most frequently recommended songs """
        from collections import Counter
        
        counts = Counter(self.recommended_music)
        most_common = counts.most_common(10)
        songs, freq = zip(*most_common)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(y=list(songs), x=list(freq), palette="viridis")
        plt.xlabel("Number of Recommendations", fontsize=14)
        plt.ylabel("Songs", fontsize=14)
        plt.title("Most Recommended Songs Over Time", fontsize=16)
        plt.show()

# Running the simulation
rec_sys = RecommenderSystem()
rec_sys.simulate()
rec_sys.plot_preferences()
rec_sys.plot_recommendation_trend()
