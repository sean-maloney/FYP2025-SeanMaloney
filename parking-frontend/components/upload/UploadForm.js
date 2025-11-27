import { useState } from "react"; //import use state hook from react so i can store and update component values

function UploadForm() { //defines the uploadform component
    const [video, setVideo] = useState(null); //stores the selected video
    const [jobId, setJobId] = useState(null); //will hold job id returned by back end after uploading
    const [message, setMessage] = useState(""); //stores success or error messages to show user
    const [isUploading, setIsUploading] = useState(false); //tracks if a upload is currently happening

    const API_BASE = "http://127.0.0.1:8000";//base url to backend api

    function fileChangeHandler(event){ //calls when user selects a file
        const file = event.target.files[0]; //gets the first file selected in file input
        setVideo(file || null); //saves the selected file into state
        setMessage(""); //clears any previous messages 
        setJobId(null); //clear previous job id results
    }

    async function submitHandler(event){ //called when the use clicks upload and process
        event.preventDefault(); //stops page from refreshing on submit

        if (!video){ //if no file was chosed it will show a message and then stop
            setMessage("Please choose a video file first.");
            return;
        }

        setIsUploading(true); //disable buttons and show uploading (marks uploading has started)
        setMessage(""); //clears old messages

        try { //start of error handling block
            const body = new FormData(); //creates a formdata object to send video file
            body.append("file", video); //adds video to upload request as file

            //sends a post to my fastapi backend to upload the file
            const response = await fetch(`${API_BASE}/api/videos`, {method: "POST", body:body,});

            if(!response.ok){ //checks if the server returned a error
                const errData = await response.json().catch(() => ({})); //try and get details of error from the response
                throw new Error(errData.detail || "Upload failed"); //if error message unreadable force an error message thats readable
            }

            const data = await response.json(); //parses the successful response JSON
            setJobId(data.job_id); //saves the job id that is returned from the back end
            setMessage(data.message || "Upload complete."); //shows a success message
            }
            catch (error) { //if anything goes wrong this block runs
                console.error(error); //logs the error for debugging
                setMessage(`Error : ${error.message}`); //displays the error to the user
            }
            finally{ //runs no matter what and re enables the upload button
                setIsUploading(false);
            }
    }

    return(
        <section>
            <form onSubmit={submitHandler} style={{marginTop: "5px"}}> {/*the upload form that triggers the submit handler*/}
                <div style={{ marginBottom: "4px"}}> {/*wrapper for spacing*/}
                    <input type="file" accept ="video/*" onChange={fileChangeHandler}/> {/*file picker for video files onli, calls file chnge handler when a file is selected*/}
                </div>
                
                <button type="submit" disabled={isUploading}> {/*the upload button, it is disabled during an upload, the tech changes based on is uploading*/}
                    {isUploading ? "Uploading..." : "Upload & Process"} 
                </button>
            </form>

                {message && <p style={{ marginTop: "5px"}}>{message}</p>} {/*if the message is not empty then show it*/}
                {jobId && ( // if a job id exists give me the link to it
                    <p style={{ marginTop: "3px"}}> {/*message + link to view jobid*/}
                        View Processed video:{" "}
                        <a href={`/view/${jobId}`}>click here</a>
                    </p>
                )}
        </section>
    );
}

export default UploadForm; //make sure the component available to import in other files