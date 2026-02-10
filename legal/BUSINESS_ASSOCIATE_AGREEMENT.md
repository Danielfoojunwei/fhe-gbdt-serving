# Business Associate Agreement (BAA)

**FHE-GBDT Privacy-Preserving ML Inference Service**

---

## HIPAA Business Associate Agreement Template

This Business Associate Agreement ("Agreement") is entered into as of **[EFFECTIVE DATE]** ("Effective Date") by and between:

**Covered Entity:**
[COVERED ENTITY NAME]
[ADDRESS]
("Covered Entity" or "Customer")

**Business Associate:**
[BUSINESS ASSOCIATE NAME]
[ADDRESS]
("Business Associate" or "Service Provider")

---

## RECITALS

**WHEREAS**, Covered Entity wishes to engage Business Associate to provide privacy-preserving machine learning inference services using Fully Homomorphic Encryption (FHE) technology;

**WHEREAS**, Business Associate's FHE-GBDT platform processes encrypted data without accessing the underlying Protected Health Information (PHI) in plaintext form;

**WHEREAS**, the Health Insurance Portability and Accountability Act of 1996 ("HIPAA") and its implementing regulations at 45 C.F.R. Parts 160 and 164 require Covered Entities to enter into agreements with their Business Associates;

**NOW, THEREFORE**, in consideration of the mutual promises and covenants herein, the parties agree as follows:

---

## ARTICLE 1: DEFINITIONS

### 1.1 HIPAA Definitions
Terms used in this Agreement shall have the same meaning as defined in 45 C.F.R. Parts 160 and 164, including but not limited to:
- **Protected Health Information (PHI)**: Individually identifiable health information transmitted or maintained in any form
- **Electronic Protected Health Information (ePHI)**: PHI transmitted or maintained in electronic form
- **Security Incident**: The attempted or successful unauthorized access, use, disclosure, modification, or destruction of information
- **Breach**: The acquisition, access, use, or disclosure of PHI in violation of the Privacy Rule

### 1.2 Service-Specific Definitions
- **FHE (Fully Homomorphic Encryption)**: Cryptographic technology enabling computation on encrypted data without decryption
- **Encrypted Feature Vectors**: Customer data encrypted using FHE before transmission to the Service
- **Ciphertext Processing**: Computation performed on encrypted data, producing encrypted results

---

## ARTICLE 2: PRIVACY-PRESERVING ARCHITECTURE

### 2.1 FHE Data Processing Model
The parties acknowledge that Business Associate's service operates using a unique privacy-preserving architecture:

a) **Client-Side Encryption**: All PHI is encrypted by Customer using FHE before transmission to Business Associate

b) **Blind Processing**: Business Associate performs machine learning inference on ciphertext only, without access to plaintext PHI

c) **Client-Side Decryption**: Only Customer possesses the decryption keys and can access inference results

d) **No Plaintext Exposure**: Business Associate's systems are designed to never have access to PHI in plaintext form

### 2.2 Technical Controls
Business Associate maintains the following technical safeguards:

| Control | Implementation |
|---------|----------------|
| Encryption in Transit | TLS 1.3 minimum, mTLS between services |
| Data Processing | FHE computation on ciphertext only |
| Key Management | Secret keys remain with Customer; Eval keys encrypted with AES-256-GCM |
| Access Control | Tenant isolation, RBAC, API key authentication |
| Audit Logging | Metadata-only logging, no PHI in logs |

---

## ARTICLE 3: OBLIGATIONS OF BUSINESS ASSOCIATE

### 3.1 Use and Disclosure of PHI
Business Associate agrees to:

a) Not use or disclose PHI other than as permitted by this Agreement or as required by law

b) Not attempt to decrypt or access Customer's encrypted PHI

c) Process only ciphertext data as instructed by Customer through the documented API

d) Not sell PHI or use it for marketing purposes

### 3.2 Safeguards
Business Associate shall implement and maintain:

a) **Administrative Safeguards**:
   - Designated Security Officer
   - Workforce training on PHI handling
   - Security incident response procedures
   - Risk assessment and management

b) **Physical Safeguards**:
   - Secure cloud infrastructure (SOC 2 Type II certified)
   - Access controls to data centers
   - Environmental controls

c) **Technical Safeguards**:
   - FHE encryption for all data processing
   - TLS 1.3 and mTLS for data transmission
   - AES-256-GCM for key storage
   - Automated security scanning
   - Continuous monitoring and alerting

### 3.3 Subcontractors
Business Associate shall:

a) Enter into written agreements with any subcontractor that creates, receives, maintains, or transmits PHI on behalf of Business Associate

b) Ensure subcontractor agreements contain the same restrictions and conditions as this Agreement

c) Notify Customer of any subcontractors with access to encrypted data

### 3.4 Security Incidents and Breaches

#### 3.4.1 Incident Notification
Business Associate shall notify Customer within **24 hours** of discovering:
- Any Security Incident involving Customer's data
- Any attempted unauthorized access to encrypted data
- Any breach of this Agreement

#### 3.4.2 Breach Assessment
Given the FHE architecture, a breach assessment shall consider:
- Whether the compromised data was in ciphertext form
- Whether decryption keys could have been accessed
- Whether any plaintext PHI was exposed

#### 3.4.3 Breach Notification
If a Breach of unsecured PHI occurs, Business Associate shall:
- Provide written notice with details required under 45 C.F.R. 164.410
- Cooperate with Customer's breach notification obligations
- Document findings and remediation actions

### 3.5 Access and Amendment
Upon Customer's request, Business Associate shall:

a) Make PHI available to satisfy Customer's obligations under 45 C.F.R. 164.524 (Access)

b) Make PHI available for amendment under 45 C.F.R. 164.526

**Note**: Due to the FHE architecture, Business Associate cannot directly access or amend plaintext PHI. Customer retains full control over their data and can access it by decrypting results locally.

### 3.6 Accounting of Disclosures
Business Associate shall:

a) Maintain audit logs of all access to Customer's encrypted data

b) Provide accounting information within **30 days** of Customer's request

c) Include: date, nature of access, and purpose for each logged event

### 3.7 Compliance Audits
Business Associate shall:

a) Allow Customer to conduct audits of Business Associate's compliance with this Agreement

b) Provide access to policies, procedures, and audit logs upon reasonable request

c) Cooperate with regulatory inquiries and investigations

### 3.8 Documentation
Business Associate shall maintain for **6 years**:

a) This Agreement and any amendments

b) Security policies and procedures

c) Risk assessments and audit reports

d) Training records

e) Incident response documentation

---

## ARTICLE 4: OBLIGATIONS OF COVERED ENTITY

### 4.1 Encryption Responsibilities
Covered Entity shall:

a) Implement FHE encryption for all PHI before transmission to Business Associate

b) Securely manage and store FHE secret keys

c) Not transmit plaintext PHI to Business Associate

### 4.2 Authorizations
Covered Entity represents that:

a) It has obtained any required authorizations or consents for PHI processing

b) It has complied with the HIPAA Privacy Rule minimum necessary standard

c) Any instructions to Business Associate will not cause Business Associate to violate HIPAA

### 4.3 Notice of Changes
Covered Entity shall notify Business Associate of:

a) Changes to its Notice of Privacy Practices affecting Business Associate's obligations

b) Any restrictions on use or disclosure of PHI

c) Any revocation of authorization by individuals

---

## ARTICLE 5: TERM AND TERMINATION

### 5.1 Term
This Agreement shall be effective as of the Effective Date and shall remain in effect until:

a) Termination by either party with **90 days** written notice

b) Termination for cause due to material breach

c) Termination of the underlying service agreement

### 5.2 Effect of Termination
Upon termination:

a) Business Associate shall return or destroy all encrypted data within **30 days**

b) If return or destruction is not feasible, Business Associate shall extend protections indefinitely

c) Business Associate shall certify in writing the return or destruction of data

### 5.3 Survival
The following provisions shall survive termination:
- Article 3.8 (Documentation retention)
- Article 5.2 (Effect of Termination)
- Article 6 (Indemnification)

---

## ARTICLE 6: INDEMNIFICATION

### 6.1 By Business Associate
Business Associate shall indemnify Covered Entity for:

a) Any breach of this Agreement by Business Associate

b) Any unauthorized use or disclosure of PHI by Business Associate

c) Any penalties or fines resulting from Business Associate's non-compliance

### 6.2 By Covered Entity
Covered Entity shall indemnify Business Associate for:

a) Any breach of this Agreement by Covered Entity

b) Any unauthorized disclosure of PHI caused by Covered Entity's failure to properly encrypt data

c) Any claims arising from Covered Entity's instructions that violated HIPAA

---

## ARTICLE 7: MISCELLANEOUS

### 7.1 Amendment
This Agreement may only be amended in writing signed by both parties. Amendment is required to comply with changes in HIPAA regulations.

### 7.2 Interpretation
Any ambiguity in this Agreement shall be interpreted to comply with HIPAA requirements.

### 7.3 No Third-Party Beneficiaries
Nothing in this Agreement creates rights in any third party.

### 7.4 Governing Law
This Agreement shall be governed by federal law (HIPAA) and the laws of **[STATE]**.

### 7.5 Entire Agreement
This Agreement constitutes the entire agreement between the parties regarding HIPAA compliance.

---

## ARTICLE 8: FHE-SPECIFIC PROVISIONS

### 8.1 Acknowledgment of Privacy Architecture
The parties acknowledge that:

a) Business Associate's FHE technology provides privacy protection that **exceeds** standard HIPAA technical safeguards

b) PHI is processed in encrypted form only, providing mathematical privacy guarantees

c) In the event of a security incident, compromised data would be ciphertext that is computationally infeasible to decrypt without Customer's secret keys

### 8.2 Breach Impact Assessment
For breach risk assessment purposes:

a) Data encrypted with FHE is considered "secured" under 45 C.F.R. 164.402(2)

b) A breach involving only ciphertext may not require notification if decryption keys were not compromised

c) Customer retains responsibility for securing FHE secret keys

### 8.3 Minimum Necessary
The FHE architecture inherently supports the minimum necessary standard by ensuring Business Associate only processes the specific encrypted features required for inference.

---

## SIGNATURES

**COVERED ENTITY:**

Signature: _________________________

Name: _________________________

Title: _________________________

Date: _________________________


**BUSINESS ASSOCIATE:**

Signature: _________________________

Name: _________________________

Title: _________________________

Date: _________________________

---

## EXHIBIT A: TECHNICAL SPECIFICATIONS

### A.1 Encryption Standards

| Component | Specification |
|-----------|---------------|
| FHE Scheme | N2HE (RLWE + LWE hybrid) |
| Security Level | 128-bit (lattice-based, post-quantum) |
| Ring Dimension | 2048/4096/8192 |
| Ciphertext Modulus | 2^32 |
| Transport Encryption | TLS 1.3 |
| Key Storage Encryption | AES-256-GCM |

### A.2 Compliance Certifications

- SOC 2 Type II
- ISO 27001:2022
- ISO 27701:2019

### A.3 Service Level Objectives

| Metric | Target |
|--------|--------|
| Availability | 99.9% |
| P95 Latency | <100ms |
| Error Rate | <0.1% |

---

## EXHIBIT B: DATA PROCESSING ACTIVITIES

| Activity | Description | PHI Exposure |
|----------|-------------|--------------|
| Feature Encryption | Customer encrypts PHI locally | None (Customer only) |
| Ciphertext Transmission | Encrypted data sent via TLS | None |
| FHE Inference | Computation on ciphertext | None |
| Result Transmission | Encrypted result returned | None |
| Result Decryption | Customer decrypts locally | None (Customer only) |

---

## EXHIBIT C: SECURITY CONTACTS

**Business Associate Security Officer:**
- Name: [SECURITY OFFICER NAME]
- Email: security@[domain].com
- Phone: [PHONE NUMBER]

**Incident Reporting:**
- Email: security-incidents@[domain].com
- Emergency: [24/7 PHONE NUMBER]

**Covered Entity Security Contact:**
- Name: _________________________
- Email: _________________________
- Phone: _________________________

---

*Document Version: 1.0*
*Last Updated: 2026-02-03*
*Review Frequency: Annual*
