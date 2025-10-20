import { useState } from "react"
import {Button} from "./components/util"
import {MovieList} from "./components/movies"
import moviesService from "./services/moviesService"
import "./styles.css"
function App() {
  let [searchString, setSearchString] = useState("")
  let [searchMovies, setSearchMovies] = useState([])
  let [selectedMovies, setSelectedMovies] = useState([])
  let [recommendedMovies, setrecommendedMovies] = useState([])

  return (
    <>
      <div>
        <input
          type="text"
          placeholder="search here"
          value={searchString}
          onChange={handleSearchStringChange}
        />
        <Button
          onClick={handleSearch}
          text="Search"/>

        <Button
          onClick={handlePredict}
          text="Predict"/>
      </div>
      <div>
        <p>Recommended Movies:</p>
        {
          recommendedMovies.length === 0
          ? "please select movies"
          : <MovieList
            className="recommended-movies"
            movies={recommendedMovies}
            onMovieClick={() => null}
            />
        }

        <p>Selected Movies:</p>
        <MovieList
          className="selected-movies"
          movies={selectedMovies}
          onMovieClick={handleSelectedMovieClick}
        />
        <p>Search Movies:</p>
        <MovieList
          className="search-movies"
          movies={searchMovies}
          onMovieClick={handleSearchMovieClick}
        />
      </div>
    </>
  )

  function handleSearchStringChange(event) {
    setSearchString(event.target.value)
  }
  function handleSearch() {
    moviesService
    .search(searchString)
    .then(response => {
      const searchResult = response.data.results
      setSearchMovies(searchResult.filter(notSelected))
    })
  }

  function notSelected(movie) {
    const selectedMoviesIds = selectedMovies.map(m => m.id)
    return !selectedMoviesIds.includes(movie.id)
  }

  function handlePredict() {
    const moviesIds = selectedMovies.map(movie => movie.id)
    moviesService
    .predict(moviesIds)
    .then(response => {
      // console.log(response.data)
      setrecommendedMovies(response.data.prediction)
    })
  }

  function handleSearchMovieClick(event, clickedMovie) {
    setSelectedMovies([...selectedMovies, clickedMovie])

    setSearchMovies(searchMovies
            .filter(movie => movie.id != clickedMovie.id))
  }
  function handleSelectedMovieClick(event, clickedMovie) {
    setSelectedMovies(selectedMovies.filter(movie => movie.id != clickedMovie.id))
  }
}

export default App
