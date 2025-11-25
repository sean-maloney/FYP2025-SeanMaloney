import Link from "next/link"
import classes from "../style/Home.module.css";

function HomePage(){
    return(
        <div className = {classes.container}>
            <h1>Parking Detector</h1>
            <p className={classes.text}>
                Simple Demo: just upload a parking lot video, the view the video ran through inference
            </p>

            <div className={classes.buttons}>
                <Link href="/upload">
                    <button className={classes.btn}>Upload Video</button>
                </Link>
                <Link href="/view/1">
                    <button className = {classes.btnSecondary}>View Output (example)</button>
                </Link>
            </div>
        </div>
    );
}

export default HomePage;