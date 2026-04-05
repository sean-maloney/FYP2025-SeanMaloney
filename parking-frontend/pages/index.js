import { useState } from "react";
import LoginScreen from "../components/app/LoginScreen";
import ParkingExperience from "../components/app/ParkingExperience";

export default function HomePage() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  if (!isLoggedIn) {
    return <LoginScreen onContinue={() => setIsLoggedIn(true)} />;
  }

  return <ParkingExperience />;
}