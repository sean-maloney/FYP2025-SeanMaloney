import { useRouter} from "next/router";

function ViewVideoPage(){
    const router = useRouter();
    const {jobId} = router.query;

    const API_BASE = "http://127.0.0.1:8000";

    if (!jobId){
        return <p style={{padding:"5px"}}>Loading...</p>;
    }
    
    return (
        <main style = {{ padding: "5px"}}>
            <h2>Processed Video</h2>
            <p>Video is being streamed directly from the FastAPI backend.</p>

            <div style = {{ marginTop: "2px"}}>
                <video
                controls
                width="800"
                style={{ maxWidth : "100%" }}
                >
                    <source src={`${API_BASE}/api/videos/${jobId}`} type="video/mp4" />
                    Browser does not support the video tag.
                </video>
            </div>
        </main>
    );
}


export default ViewVideoPage;