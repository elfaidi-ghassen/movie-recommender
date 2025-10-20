from flask import Flask, request, jsonify
from flask_cors import CORS
from main import predict, search_films
from dotenv import load_dotenv
import os
import requests
load_dotenv()  # Loads variables from .env file
TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
# so the dotenv library basically loads the env file
# then parse its content, and mutate the os.environ object
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500'
app = Flask(__name__)
CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/api/predict", methods=["POST"])
def get_data():
    data = request.get_json()
    recommendations = [{"id": id, "title": title, "imdb_id":imdb_id} for id, title, imdb_id in predict(data)]
    # print(recommendations)
    
    return jsonify({"prediction": recommendations})
    # return jsonify({"prediction": [{"id": 72, "imdb_id": '0113537', "title": 'Kicking and Screaming (1995)'}]})

@app.route("/api/search/<q>", methods=["GET"])
def search(q):
    results = [{"id": id, "title": title, "imdb_id":imdb_id}
                for id, title, imdb_id in search_films(q)]
    results = list(map(add_image, results))
    return {"results": results}

def add_image(movie):
    imageURL = get_image(f"tt{movie["imdb_id"]}")
    return {**movie, "imgURL": imageURL}


image_cache = {}
def get_image(imdb_id):

    if imdb_id in image_cache:
        return image_cache[imdb_id]
    try:
        find_url = f"{TMDB_BASE_URL}/find/{imdb_id}"
        params = {
            "api_key":TMDB_API_KEY,
            'external_source': 'imdb_id'
        }
        response = requests.get(find_url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get('movie_results') and len(data['movie_results']) > 0:
            poster_path = data['movie_results'][0].get('poster_path')
            if poster_path:
                image_cache[imdb_id] = f'{TMDB_IMAGE_BASE_URL}{poster_path}'
                return f'{TMDB_IMAGE_BASE_URL}{poster_path}'
        return None  # Return None if no image found
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image for IMDb ID {imdb_id}: {e}")
        return None



if __name__ == "__main__":
    app.run(debug=True)