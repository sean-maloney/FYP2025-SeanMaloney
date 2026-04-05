import { useState } from "react";
import styles from "../../styles/ParkingExperience.module.css";

export default function LoginScreen({ onContinue }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  function handleSubmit(event) {
    event.preventDefault();
    onContinue({ email, password });
  }

  return (
    <div className={styles.loginShell}>
      <div className={styles.loginCard}>
        <div className={styles.brandRow}>
          <div className={styles.brandBadge}>P</div>
          <div>
            <div className={styles.brandTitle}>Parking Finder</div>
            <div className={styles.brandSubtitle}>Smart parking guidance</div>
          </div>
        </div>

        <h1 className={styles.loginTitle}>Welcome Back</h1>
        <p className={styles.loginText}>Login to continue</p>

        <form onSubmit={handleSubmit} className={styles.formStack}>
          <input
            className={styles.input}
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />

          <input
            className={styles.input}
            placeholder="Password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />

          <button type="submit" className={styles.primaryButton}>
            Continue
          </button>
        </form>
      </div>
    </div>
  );
}