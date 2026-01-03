import { request } from "express";
import { ValidateNewUserBody } from "./types"

export const validateNewUserBody = (requestBody: ValidateNewUserBody,otherStrings: string[]): boolean => {
    if (typeof (requestBody.userName) !== 'string' )
        return false;
    if (requestBody.userName.length < 1)
        return false;
    if (otherStrings.includes(requestBody.userName))
        return true;
}