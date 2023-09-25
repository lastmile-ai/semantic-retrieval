/**
 * Abstract class for loading string content from a file.
 */
export abstract class BaseFileLoader {
    path: string;
  
    constructor(path: string) {
      this.path = path;
    }
  
    abstract loadContent(): Promise<string>;
  }