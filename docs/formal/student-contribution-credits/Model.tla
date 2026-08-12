------------------------------ MODULE Model -------------------------------
EXTENDS Integers, Naturals, FiniteSets, TLC

CONSTANTS
    StudentA, StudentB,
    StrokeA1, StrokeA2, PhotoB1,
    PageA, PhotoPageB, ActiveSources,
    MaxAttempts, MaxTarget, DailyCap

Students == {StudentA, StudentB}
Sources == {StrokeA1, StrokeA2, PhotoB1}
Pages == {PageA, PhotoPageB}
Policies == {1, 2}

StudentOf == [source \in Sources |->
    IF source = PhotoB1 THEN StudentB ELSE StudentA]

PageOf == [source \in Sources |->
    CASE source \in {StrokeA1, StrokeA2} -> PageA
      [] OTHER -> PhotoPageB]

SourceKinds == [source \in Sources |->
    IF source = PhotoB1 THEN "photo" ELSE "stroke"]

SourceStates == {"absent", "complete"}
JobStates == {"none", "pending", "processing", "retry", "completed", "failed"}
Decisions == {"unknown", "accepted", "rejected"}
Judgments == {"none", "accepted", "rejected"}

(***************************************************************************
The source process abstracts a tenant-scoped API writer, reconciler, and
atomic Mongo claim for one immutable source version. Separate source
processes interleave. A retry after lease expiry represents another worker
winning the next compare-and-set claim. Judgment and ledger persistence are
separate labels so a crash boundary exists between them.
***************************************************************************)

(* --fair algorithm CreditsLifecycle
variables
    currentPolicy = 1,
    policyChanged = FALSE,
    sourceState = [source \in Sources |-> "absent"],
    normalPipeline = [source \in Sources |-> "absent"],
    jobState = [source \in Sources |-> "none"],
    jobPolicy = [source \in Sources |-> 0],
    attempts = [source \in Sources |-> 0],
    leaseEpoch = [source \in Sources |-> 0],
    staleEpoch = [source \in Sources |-> 0],
    decision = [source \in Sources |-> "unknown"],
    targetCredits = [source \in Sources |-> 0],
    judgment = [source \in Sources |-> "none"],
    judgmentPolicy = [source \in Sources |-> 0],
    ledgerCommitted = [source \in Sources |-> FALSE],
    ledger = [source \in Sources |-> 0],
    pageAward = [page \in Pages |-> 0],
    studentTotal = [student \in Students |-> 0];

process SourceActor \in ActiveSources
begin
SourceLoop:
    while TRUE do
        if sourceState[self] = "absent" then
            \* Primary pipeline completes independently. Enqueue may be
            \* missed here; the next iteration represents reconciliation.
            sourceState[self] := "complete" ||
            normalPipeline[self] := "complete";
        elsif jobState[self] = "none" then
            jobState[self] := "pending" ||
            jobPolicy[self] := currentPolicy;
        elsif jobState[self] = "pending" \/ jobState[self] = "retry" then
            jobState[self] := "processing" ||
            attempts[self] := attempts[self] + 1 ||
            leaseEpoch[self] := leaseEpoch[self] + 1 ||
            decision[self] := "unknown" ||
            targetCredits[self] := 0;
        elsif jobState[self] = "processing" /\ decision[self] = "unknown" then
            either
                \* Deterministic and semantic gates accept a bounded target.
                with target \in 1..MaxTarget do
                    decision[self] := "accepted" ||
                    targetCredits[self] := target;
                end with;
            or
                \* A bad-quality result rejects credits only.
                decision[self] := "rejected" ||
                targetCredits[self] := 0;
            or
                \* Transient dependency failure is never quality rejection.
                if attempts[self] >= MaxAttempts then
                    jobState[self] := "failed";
                else
                    jobState[self] := "retry";
                end if;
            or
                \* Worker dies and leaves a stale lease token.
                staleEpoch[self] := leaseEpoch[self] ||
                jobState[self] := IF attempts[self] >= MaxAttempts
                                      THEN "failed"
                                      ELSE "retry";
            end either;
        elsif jobState[self] = "processing"
              /\ decision[self] # "unknown"
              /\ judgment[self] = "none" then
            PersistJudgment:
            judgment[self] := decision[self] ||
            judgmentPolicy[self] := jobPolicy[self];
        elsif judgment[self] # "none" /\ ~ledgerCommitted[self] then
            PersistLedger:
            with pageDelta = IF judgment[self] = "accepted"
                                THEN IF targetCredits[self] > pageAward[PageOf[self]]
                                        THEN targetCredits[self] - pageAward[PageOf[self]]
                                        ELSE 0
                                ELSE 0,
                 remainingDaily = DailyCap - studentTotal[StudentOf[self]] do
            with delta = IF pageDelta > remainingDaily THEN remainingDaily ELSE pageDelta
            do
                ledger[self] := delta ||
                pageAward[PageOf[self]] := pageAward[PageOf[self]] + delta ||
                studentTotal[StudentOf[self]] := studentTotal[StudentOf[self]] + delta ||
                ledgerCommitted[self] := TRUE;
            end with;
            end with;
        elsif ledgerCommitted[self] /\ jobState[self] # "completed" then
            FinishJob:
            jobState[self] := "completed";
        else
            skip;
        end if;
    end while;
end process;

process PolicyAdmin = "policy-admin"
begin
PolicyLoop:
    if ~policyChanged then
        currentPolicy := 2 ||
        policyChanged := TRUE;
    end if;
end process;
end algorithm; *)
\* BEGIN TRANSLATION (chksum(pcal) = "271f8b91" /\ chksum(tla) = "973e7b4a")
VARIABLES currentPolicy, policyChanged, sourceState, normalPipeline, jobState, 
          jobPolicy, attempts, leaseEpoch, staleEpoch, decision, 
          targetCredits, judgment, judgmentPolicy, ledgerCommitted, ledger, 
          pageAward, studentTotal, pc

vars == << currentPolicy, policyChanged, sourceState, normalPipeline, 
           jobState, jobPolicy, attempts, leaseEpoch, staleEpoch, decision, 
           targetCredits, judgment, judgmentPolicy, ledgerCommitted, ledger, 
           pageAward, studentTotal, pc >>

ProcSet == (ActiveSources) \cup {"policy-admin"}

Init == (* Global variables *)
        /\ currentPolicy = 1
        /\ policyChanged = FALSE
        /\ sourceState = [source \in Sources |-> "absent"]
        /\ normalPipeline = [source \in Sources |-> "absent"]
        /\ jobState = [source \in Sources |-> "none"]
        /\ jobPolicy = [source \in Sources |-> 0]
        /\ attempts = [source \in Sources |-> 0]
        /\ leaseEpoch = [source \in Sources |-> 0]
        /\ staleEpoch = [source \in Sources |-> 0]
        /\ decision = [source \in Sources |-> "unknown"]
        /\ targetCredits = [source \in Sources |-> 0]
        /\ judgment = [source \in Sources |-> "none"]
        /\ judgmentPolicy = [source \in Sources |-> 0]
        /\ ledgerCommitted = [source \in Sources |-> FALSE]
        /\ ledger = [source \in Sources |-> 0]
        /\ pageAward = [page \in Pages |-> 0]
        /\ studentTotal = [student \in Students |-> 0]
        /\ pc = [self \in ProcSet |-> CASE self \in ActiveSources -> "SourceLoop"
                                        [] self = "policy-admin" -> "PolicyLoop"]

SourceLoop(self) == /\ pc[self] = "SourceLoop"
                    /\ IF sourceState[self] = "absent"
                          THEN /\ /\ normalPipeline' = [normalPipeline EXCEPT ![self] = "complete"]
                                  /\ sourceState' = [sourceState EXCEPT ![self] = "complete"]
                               /\ pc' = [pc EXCEPT ![self] = "SourceLoop"]
                               /\ UNCHANGED << jobState, jobPolicy, attempts, 
                                               leaseEpoch, staleEpoch, 
                                               decision, targetCredits >>
                          ELSE /\ IF jobState[self] = "none"
                                     THEN /\ /\ jobPolicy' = [jobPolicy EXCEPT ![self] = currentPolicy]
                                             /\ jobState' = [jobState EXCEPT ![self] = "pending"]
                                          /\ pc' = [pc EXCEPT ![self] = "SourceLoop"]
                                          /\ UNCHANGED << attempts, leaseEpoch, 
                                                          staleEpoch, decision, 
                                                          targetCredits >>
                                     ELSE /\ IF jobState[self] = "pending" \/ jobState[self] = "retry"
                                                THEN /\ /\ attempts' = [attempts EXCEPT ![self] = attempts[self] + 1]
                                                        /\ decision' = [decision EXCEPT ![self] = "unknown"]
                                                        /\ jobState' = [jobState EXCEPT ![self] = "processing"]
                                                        /\ leaseEpoch' = [leaseEpoch EXCEPT ![self] = leaseEpoch[self] + 1]
                                                        /\ targetCredits' = [targetCredits EXCEPT ![self] = 0]
                                                     /\ pc' = [pc EXCEPT ![self] = "SourceLoop"]
                                                     /\ UNCHANGED staleEpoch
                                                ELSE /\ IF jobState[self] = "processing" /\ decision[self] = "unknown"
                                                           THEN /\ \/ /\ \E target \in 1..MaxTarget:
                                                                           /\ decision' = [decision EXCEPT ![self] = "accepted"]
                                                                           /\ targetCredits' = [targetCredits EXCEPT ![self] = target]
                                                                      /\ UNCHANGED <<jobState, staleEpoch>>
                                                                   \/ /\ /\ decision' = [decision EXCEPT ![self] = "rejected"]
                                                                         /\ targetCredits' = [targetCredits EXCEPT ![self] = 0]
                                                                      /\ UNCHANGED <<jobState, staleEpoch>>
                                                                   \/ /\ IF attempts[self] >= MaxAttempts
                                                                            THEN /\ jobState' = [jobState EXCEPT ![self] = "failed"]
                                                                            ELSE /\ jobState' = [jobState EXCEPT ![self] = "retry"]
                                                                      /\ UNCHANGED <<staleEpoch, decision, targetCredits>>
                                                                   \/ /\ /\ jobState' = [jobState EXCEPT ![self] = IF attempts[self] >= MaxAttempts
                                                                                                                       THEN "failed"
                                                                                                                       ELSE "retry"]
                                                                         /\ staleEpoch' = [staleEpoch EXCEPT ![self] = leaseEpoch[self]]
                                                                      /\ UNCHANGED <<decision, targetCredits>>
                                                                /\ pc' = [pc EXCEPT ![self] = "SourceLoop"]
                                                           ELSE /\ IF jobState[self] = "processing"
                                                                      /\ decision[self] # "unknown"
                                                                      /\ judgment[self] = "none"
                                                                      THEN /\ pc' = [pc EXCEPT ![self] = "PersistJudgment"]
                                                                      ELSE /\ IF judgment[self] # "none" /\ ~ledgerCommitted[self]
                                                                                 THEN /\ pc' = [pc EXCEPT ![self] = "PersistLedger"]
                                                                                 ELSE /\ IF ledgerCommitted[self] /\ jobState[self] # "completed"
                                                                                            THEN /\ pc' = [pc EXCEPT ![self] = "FinishJob"]
                                                                                            ELSE /\ TRUE
                                                                                                 /\ pc' = [pc EXCEPT ![self] = "SourceLoop"]
                                                                /\ UNCHANGED << jobState, 
                                                                                staleEpoch, 
                                                                                decision, 
                                                                                targetCredits >>
                                                     /\ UNCHANGED << attempts, 
                                                                     leaseEpoch >>
                                          /\ UNCHANGED jobPolicy
                               /\ UNCHANGED << sourceState, normalPipeline >>
                    /\ UNCHANGED << currentPolicy, policyChanged, judgment, 
                                    judgmentPolicy, ledgerCommitted, ledger, 
                                    pageAward, studentTotal >>

PersistJudgment(self) == /\ pc[self] = "PersistJudgment"
                         /\ /\ judgment' = [judgment EXCEPT ![self] = decision[self]]
                            /\ judgmentPolicy' = [judgmentPolicy EXCEPT ![self] = jobPolicy[self]]
                         /\ pc' = [pc EXCEPT ![self] = "SourceLoop"]
                         /\ UNCHANGED << currentPolicy, policyChanged, 
                                         sourceState, normalPipeline, jobState, 
                                         jobPolicy, attempts, leaseEpoch, 
                                         staleEpoch, decision, targetCredits, 
                                         ledgerCommitted, ledger, pageAward, 
                                         studentTotal >>

PersistLedger(self) == /\ pc[self] = "PersistLedger"
                       /\ LET pageDelta == IF judgment[self] = "accepted"
                                              THEN IF targetCredits[self] > pageAward[PageOf[self]]
                                                      THEN targetCredits[self] - pageAward[PageOf[self]]
                                                      ELSE 0
                                              ELSE 0 IN
                            LET remainingDaily == DailyCap - studentTotal[StudentOf[self]] IN
                              LET delta == IF pageDelta > remainingDaily THEN remainingDaily ELSE pageDelta IN
                                /\ ledger' = [ledger EXCEPT ![self] = delta]
                                /\ ledgerCommitted' = [ledgerCommitted EXCEPT ![self] = TRUE]
                                /\ pageAward' = [pageAward EXCEPT ![PageOf[self]] = pageAward[PageOf[self]] + delta]
                                /\ studentTotal' = [studentTotal EXCEPT ![StudentOf[self]] = studentTotal[StudentOf[self]] + delta]
                       /\ pc' = [pc EXCEPT ![self] = "SourceLoop"]
                       /\ UNCHANGED << currentPolicy, policyChanged, 
                                       sourceState, normalPipeline, jobState, 
                                       jobPolicy, attempts, leaseEpoch, 
                                       staleEpoch, decision, targetCredits, 
                                       judgment, judgmentPolicy >>

FinishJob(self) == /\ pc[self] = "FinishJob"
                   /\ jobState' = [jobState EXCEPT ![self] = "completed"]
                   /\ pc' = [pc EXCEPT ![self] = "SourceLoop"]
                   /\ UNCHANGED << currentPolicy, policyChanged, sourceState, 
                                   normalPipeline, jobPolicy, attempts, 
                                   leaseEpoch, staleEpoch, decision, 
                                   targetCredits, judgment, judgmentPolicy, 
                                   ledgerCommitted, ledger, pageAward, 
                                   studentTotal >>

SourceActor(self) == SourceLoop(self) \/ PersistJudgment(self)
                        \/ PersistLedger(self) \/ FinishJob(self)

PolicyLoop == /\ pc["policy-admin"] = "PolicyLoop"
              /\ IF ~policyChanged
                    THEN /\ /\ currentPolicy' = 2
                            /\ policyChanged' = TRUE
                    ELSE /\ TRUE
                         /\ UNCHANGED << currentPolicy, policyChanged >>
              /\ pc' = [pc EXCEPT !["policy-admin"] = "Done"]
              /\ UNCHANGED << sourceState, normalPipeline, jobState, jobPolicy, 
                              attempts, leaseEpoch, staleEpoch, decision, 
                              targetCredits, judgment, judgmentPolicy, 
                              ledgerCommitted, ledger, pageAward, studentTotal >>

PolicyAdmin == PolicyLoop

Next == PolicyAdmin
           \/ (\E self \in ActiveSources: SourceActor(self))

Spec == /\ Init /\ [][Next]_vars
        /\ WF_vars(Next)

\* END TRANSLATION 

RECURSIVE StudentLedgerSum(_, _)
StudentLedgerSum(remaining, student) ==
    IF remaining = {} THEN 0
    ELSE LET source == CHOOSE source \in remaining : TRUE
         IN  (IF StudentOf[source] = student THEN ledger[source] ELSE 0)
             + StudentLedgerSum(remaining \ {source}, student)

RECURSIVE PageLedgerSum(_, _)
PageLedgerSum(remaining, page) ==
    IF remaining = {} THEN 0
    ELSE LET source == CHOOSE source \in remaining : TRUE
         IN  (IF PageOf[source] = page THEN ledger[source] ELSE 0)
             + PageLedgerSum(remaining \ {source}, page)

TypeOK ==
    /\ ActiveSources \in (SUBSET Sources) \ {{} }
    /\ currentPolicy \in Policies
    /\ policyChanged \in BOOLEAN
    /\ sourceState \in [Sources -> SourceStates]
    /\ normalPipeline \in [Sources -> SourceStates]
    /\ jobState \in [Sources -> JobStates]
    /\ jobPolicy \in [Sources -> (Policies \cup {0})]
    /\ attempts \in [Sources -> 0..MaxAttempts]
    /\ leaseEpoch \in [Sources -> 0..MaxAttempts]
    /\ staleEpoch \in [Sources -> 0..MaxAttempts]
    /\ decision \in [Sources -> Decisions]
    /\ targetCredits \in [Sources -> 0..MaxTarget]
    /\ judgment \in [Sources -> Judgments]
    /\ judgmentPolicy \in [Sources -> (Policies \cup {0})]
    /\ ledgerCommitted \in [Sources -> BOOLEAN]
    /\ ledger \in [Sources -> 0..MaxTarget]
    /\ pageAward \in [Pages -> 0..(Cardinality(Sources) * MaxTarget)]
    /\ studentTotal \in [Students -> 0..DailyCap]

NoCreditBeforeSource ==
    \A source \in Sources : ledger[source] > 0 => sourceState[source] = "complete"

PipelineIsIndependent ==
    \A source \in Sources : normalPipeline[source] = sourceState[source]

JobRequiresSource ==
    \A source \in Sources : jobState[source] # "none" => sourceState[source] = "complete"

RejectedHasNoAward ==
    \A source \in Sources : judgment[source] = "rejected" => ledger[source] = 0

CommittedJudgmentIsPinned ==
    \A source \in Sources :
        judgment[source] # "none" => judgmentPolicy[source] = jobPolicy[source]

CompletedHasCommittedLedger ==
    \A source \in Sources : jobState[source] = "completed" => ledgerCommitted[source]

TotalsMatchLedger ==
    \A student \in Students : studentTotal[student] = StudentLedgerSum(Sources, student)

DailyCapRespected ==
    \A student \in Students : studentTotal[student] <= DailyCap

PageAwardsMatchLedger ==
    \A page \in Pages : pageAward[page] = PageLedgerSum(Sources, page)

TerminalOutcomeIsVisible ==
    \A source \in Sources :
        jobState[source] = "completed" => judgment[source] # "none"

AllSourcesEventuallyTerminal ==
    <> (\A source \in ActiveSources : jobState[source] \in {"completed", "failed"})

=============================================================================
