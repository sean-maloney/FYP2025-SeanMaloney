import { useState } from "react";

function UploadForm() {
    const [video, setVideo] = useState(null);
    const [jobId, setJobId] = useState(null);
    const [message, setMessage] = useState("");
    const [isUploading, setIsUploading] = useState(false);

    const API_BASE = "http://127.0.0.1:8000";

    function fileChangeHandler(event){
        const file = event.target.files[0];
        setVideo(file || null);
        setMessage("");
        setJobId(null);
    }

    async function submitHandler(event){
        event.preventDefault();

        if (!video){
            setMessage("Please choose a video file first.");
            return;
        }

        setIsUploading(true);
        setMessage("");

        try {
            const body = new FormData();
            body.append("file", video);

            const response = await fetch(`${API_BASE}/api/videos`, {method: "POST", body:body,});

            if(!response.ok){
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || "Upload failed");
            }

            const data = await response.json();
            setJobId(data.job_id);
            setMessage(data.message || "Upload complete.");
            }
            catch (error) {
                console.error(error);
                setMessage(`Error : ${error.message}`);
            }
            finally{
                setIsUploading(false);
            }
    }

    return(
        <section>
            <form onSubmit={submitHandler} style={{marginTop: "5px"}}>
                <div style={{ marginBottom: "4px"}}>
                    <input type="file" accept ="video/*" onChange={fileChangeHandler}/>
                </div>
                
                <button type="submit" disabled={isUploading}>
                    {isUploading ? "Uploading..." : "Upload & Process"} 
                </button>
            </form>

                {message && <p style={{ marginTop: "5px"}}>{message}</p>}
                {jobID && (
                    <p style={{ marginTop: "3px"}}>
                        View Processed video:{" "}
                        <a href={`/view/${jobId}`}>click here</a>
                    </p>
                )}
        </section>
    );
}

export default UploadForm;