import { JSX, useState } from "react";
import UserNameModal from "./UserNameModal";

const SplashScreen = (): JSX.Element => {
  const [name, setName] = useState<string>("");
  const [inModal, setInModal] = useState<boolean>(true);
  return (
    <div
      style={{
        display: "flex",
        width: "100vw",
        height: "100vh",
        backgroundColor: "green",
      }}
    >
      <UserNameModal
        nameVal={name}
        setName={setName}
        open={inModal}
        setOpen={setInModal}
      />
    </div>
  );
};

export default SplashScreen;
