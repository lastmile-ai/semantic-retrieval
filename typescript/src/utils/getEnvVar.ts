// Using this helper allows us to mock environment variables in tests
export default function getEnvVar(name: string) {
  return process.env[name];
}
