import axios from "axios"
const baseURL = "http://127.0.0.1:5000/api"
function predict(moviesIds) {
    return axios
           .post(`${baseURL}/predict`, moviesIds)
}
function search(searchString) {
    return axios
           .get(`${baseURL}/search/${searchString}`)
}
export default { predict, search }
