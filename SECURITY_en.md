# Security Statement

## Running User Recommendations

For security reasons, do not use root or other administrator accounts to execute any commands. Follow the principle of least privilege.

## File Permission Control

- Set the umask value of the running system to 0027 or higher on the host (including the host machine) and in the container, ensuring that the default maximum permission of newly created directories is 750 and the default maximum permission of newly created files is 640.
- Implement permission control and other security measures for sensitive content such as personal privacy data, business assets, source files, and various files saved during operator development. For example, for permission control of the project installation directory and input public data files, refer to [A-Recommended Maximum Permission Control for Files (Directories) in Various Scenarios](#a-recommended-maximum-permission-control-for-files-directories-in-various-scenarios).
- During operator running, operator compilation files may be cached in the `kernel_meta_*` folder under the running directory to accelerate subsequent operator calls. You can perform permission control on the generated files as needed.
- Implement permission control during installation and use. For file permission reference, refer to [A-Recommended Maximum Permission Control for Files (Directories) in Various Scenarios](#a-recommended-maximum-permission-control-for-files-directories-in-various-scenarios) for settings.

## Build Security Statement

When compiling and installing this project from source, you need to compile it yourself. Some intermediate files are generated during the compilation process. After compilation, implement permission control on the intermediate files to ensure file security.

## Running Security Statement

- Prepare operator calling scripts based on the running environment resource status. If the operator calling script does not match the resource status, such as the space used for generating input data or benchmark calculation results exceeding the memory capacity limit, or the data saved locally by the script exceeding the disk space size, errors may occur and the process may exit unexpectedly.
- When an operator runs abnormally, the process exits and an error message is printed. Locate the specific error cause based on the error message, including setting synchronous operator execution and viewing log files.
- When operators are called through [PyTorch](https://gitee.com/ascend/pytorch), running errors may occur due to version mismatch. For details, refer to the [PyTorch Security Statement](https://gitee.com/ascend/pytorch#%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E).

## Public Network Address Statement

The public network addresses included in the project code are declared as follows:

| Type | Open Source Code Address | File Name                                 | Public IP Address/Public URL Address/Domain Name/Email Address/Compressed File Address                                                                                                                       | Purpose                                  |
| :--: | :----------: | :------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------- |
| Dependency |    Not applicable    | cmake/third_party/makeself-fetch.cmake | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) | Download makeself source code from gitcode as a compilation dependency   |

---

## Vulnerability Mechanism Description

[Vulnerability Management](https://gitcode.com/cann/community/blob/master/security/security.md)

## Appendix

### A-Recommended Maximum Permission Control for Files (Directories) in Various Scenarios

| Type                               | Linux Permission Reference Maximum Value |
| ---------------------------------- | ------------------- |
| User home directory                         | 750 (rwxr-x---)    |
| Program files (including script files, library files, and so on)   | 550 (r-xr-x---)    |
| Program file directory                       | 550 (r-xr-x---)    |
| Configuration file                           | 640 (rw-r-----)    |
| Configuration file directory                       | 750 (rwxr-x---)    |
| Log file (recording completed or archived)   | 440 (r--r-----)    |
| Log file (being recorded)               | 640 (rw-r-----)    |
| Log file directory                       | 750 (rwxr-x---)    |
| Debug file                          | 640 (rw-r-----)    |
| Debug file directory                      | 750 (rwxr-x---)    |
| Temporary file directory                   | 750 (rwxr-x---)    |
| Maintenance and upgrade file directory                   | 770 (rwxrwx---)    |
| Business data file                       | 640 (rw-r-----)    |
| Business data file directory                   | 750 (rwxr-x---)    |
| Key component, private key, certificate, ciphertext file directory | 700 (rwx------)    |
| Key component, private key, certificate, encrypted ciphertext     | 600 (rw-------)    |
| Encryption and decryption interface, encryption and decryption script             | 500 (r-x------)    |"
