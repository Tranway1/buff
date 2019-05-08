/*
 * Would like to implement variant of signals that uses 
 * tokio and futures. I think this could be a strong way of 
 * creating an implementation that only runs when packets need
 * to be collected, allowing the OS to dynamically schedule 
 * threads more effectively for querying, reading, and compressing.
 * It seems pretty doable with chunks and then running a tokio
 * core in a seperate thread. 
 */

 