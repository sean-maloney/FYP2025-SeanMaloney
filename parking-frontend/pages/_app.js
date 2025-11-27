import "../styles/global.css"; //loads global stylesheet into app, everywhere, can only be imported in _app.js

function MyApp({ Component, pageProps}){ // defines my app component, next.js uses this file to initialize every page
    //component = page being rendered, pageProps = any props the page needs
    return <Component {...pageProps} />; //render the active page
}

export default MyApp; //make my app available for Next.js to use as the central wrapped around whole website