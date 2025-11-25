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
    }
}