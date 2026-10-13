# Music Recommender System Simulation

### `music_recommender_simulation.py`
A simulated **content-based recommender system** for music, built from scratch with NumPy (no ML library dependency).

**How it works:**
- Generates a synthetic catalog of 50 songs, each described by 10 real audio features (Acousticness, Danceability, Energy, Instrumentalness, Liveness, Loudness, Speechiness, Valence, Tempo, Popularity)
- Initializes a random, normalized user preference vector
- `recommend()`: scores every song via dot product with the user's preference vector and returns the best match
- `update_preferences()`: after each recommendation, simulates user feedback (mostly positive, some neutral/negative) and nudges the preference vector toward or away from that song's features
- Runs 50 iterations of recommend → feedback → update, then plots (1) how each of the 10 preference dimensions evolves over time and (2) which songs were recommended most often

**Purpose:** a compact illustration of how a content-based recommender converges on a user's taste through implicit feedback, useful as a teaching tool or a starting point for a more realistic system (e.g. swapping in real track features from a music API and real interaction data).

**Requirements:** `numpy`, `matplotlib`, `seaborn`.

**Run:** `python music_recommender_simulation.py`
