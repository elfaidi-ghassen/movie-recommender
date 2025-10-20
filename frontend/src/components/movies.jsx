const Movie = ({movie, onMovieClick}) => {
  return (
    <div className="movie"
      onClick={(event) => onMovieClick(event, movie)}>
        {/* {movie.title} */}
        <img src={movie.imgURL} alt={movie.title} title={movie.title}/>
    </div>
  )
}
const MovieList = ({movies, onMovieClick, className}) => {
  return (
    <div className={className}>
      {
        movies.map(movie => {
          return (
            <Movie
              movie={movie}
              onMovieClick={onMovieClick}
              key={movie.imdb_id}/>
          )
        })
      }
    </div>
  )
}

export {Movie, MovieList}