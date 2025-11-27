import UploadForm from "../../components/upload/UploadForm"; //imports uploadform from component

function UploadPage() { //creates react component called Upload page (when someone visits /upload)
    return( 
        <main style ={{ padding: "24px"}}> {/*Renders main element and applies inline padding od 24px*/}
            <h2>Upload Video</h2> {/*Heading for page*/}
            <UploadForm /> {/*Renders the imported upload form component, where user interacts with upload interface*/}
        </main>
    );
} //closes the main and return

export default UploadPage; //makes this the default export so Next.js can load it as /upload