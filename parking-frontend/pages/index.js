import Link from "next/link" //imports next.js's built in link component that lets you navigate between pages without reloading
import classes from "../style/Home.module.css"; //imports styling classes becomes a object where each key is a css class name scoped only to the component

function HomePage(){ //declares a function name HopePage (component)
    return( //begins the jsx returned by the componennt (whatever gets rendered on screen)
        <div className = {classes.container}> {/*renders a div and applied container css class from css module*/}
            <h1>Parking Detector</h1> {/*Displays a large page title*/}
            <p className={classes.text}> {/*Renders a paragraph within text css class*/}
                Simple Demo: just upload a parking lot video, the view the video ran through inference
            </p> {/*end paragraph*/}

            <div className={classes.buttons}> {/*A wrapper div for the two buttons*/}
                <Link href="/upload"> {/*creates a link to the /upload page but without page reload*/}
                    <button className={classes.btn}>Upload Video</button> {/*Render styled button, sends user to upload page*/}
                </Link> {/*Close link*/}
                <Link href="/view/1"> {/*creates a link to /view/1*/}
                    <button className = {classes.btnSecondary}>View Output (example)</button> {/*renders another button with secondary style varaiant*/}
                </Link>{/*Closes the link*/}
            </div>
        </div>
    );
}//closes the button divs and main container div and the return statement

export default HomePage; //makes homepage component available for import elsewhere.