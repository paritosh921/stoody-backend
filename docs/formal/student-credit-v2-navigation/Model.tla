-------------------------- MODULE Model --------------------------
EXTENDS Integers, Naturals, FiniteSets, Sequences, TLC

CONSTANTS OldSource, NewSource, ActiveSources,
          MaxAttempts, MaxCompletionLookups,
          LegacyTargetCap, V2TargetCap,
          LegacyDailyCap, V2DailyCap

Sources == ActiveSources
Policies == {"legacy", "v2"}
JobStates == {"none", "pending", "processing", "retry", "completed", "failed"}
TerminalStates == {"completed", "failed"}
OpenStates == {"pending", "processing", "retry"}
Decisions == {"unknown", "accepted", "rejected", "unrecoverable"}
Surfaces == {"none", "avatar", "hamburger"}
Routes == {"home", "profile", "credits", "rewards"}

Min(a, b) == IF a <= b THEN a ELSE b
Max(a, b) == IF a >= b THEN a ELSE b

TargetCap(policy) == IF policy = "v2" THEN V2TargetCap ELSE LegacyTargetCap
DayCap(policy) == IF policy = "v2" THEN V2DailyCap ELSE LegacyDailyCap

StrokeAward(mm) == Min(5, (mm + 249) \div 250)
PhotoAward(pages) == Min(10, pages)

TierOf(total) ==
    CASE total >= 4000 -> "Luminary"
      [] total >= 1500 -> "Beacon"
      [] total >= 500 -> "Pathfinder"
      [] total >= 100 -> "Scribe"
      [] OTHER -> "Seed"

StudentMenu == <<"Profile", "Learning Credits", "Rewards", "Sign out">>
TutorMenu == <<"My Profile", "Learning Credits", "Rewards", "Leaderboard", "Sign out">>

(***************************************************************************
The model composes two bounded abstractions:

1. immutable upload sources become policy-snapshotted jobs and terminate by
   judgment, bounded worker failure, or bounded missing-completion recovery;
2. a mobile header exposes mutually exclusive avatar/hamburger surfaces and
   commits hamburger route requests in a separate step.

An atomic Enqueue or Activate label represents the tenant policy-transition
lock used by the implementation. Historical ledger values never change.
***************************************************************************)

(* --fair algorithm CreditV2Navigation
variables
    activePolicy = "legacy",
    activationRequested = FALSE,
    sourceState = [s \in Sources |-> "absent"],
    jobState = [s \in Sources |-> "none"],
    jobPolicy = [s \in Sources |-> "none"],
    completionKnown = [s \in Sources |-> FALSE],
    completionLookups = [s \in Sources |-> 0],
    attempts = [s \in Sources |-> 0],
    decision = [s \in Sources |-> "unknown"],
    ledger = [s \in Sources |-> 0],
    balance = 0,
    dailyAward = 0,
    surface = "none",
    route = "home",
    requestedRoute = "none";

fair process SourceActor \in Sources
begin
SourceComplete:
    sourceState[self] := "complete";
Enqueue:
    \* Atomic with respect to Activate: a job captures one coherent preset.
    jobState[self] := "pending" ||
    jobPolicy[self] := activePolicy;
RecoverCompletion:
    while ~completionKnown[self] /\ jobState[self] \notin TerminalStates do
        either
            completionKnown[self] := TRUE;
        or
            completionLookups[self] := completionLookups[self] + 1;
            if completionLookups[self] >= MaxCompletionLookups then
                jobState[self] := "failed" ||
                decision[self] := "unrecoverable";
            end if;
        end either;
    end while;
Claim:
    if jobState[self] \notin TerminalStates then
        if attempts[self] >= MaxAttempts then
            jobState[self] := "failed";
        else
            jobState[self] := "processing" ||
            attempts[self] := attempts[self] + 1;
        end if;
    end if;
Judge:
    if jobState[self] = "processing" then
        either
            with target \in 1..TargetCap(jobPolicy[self]) do
                ledger[self] := Min(target, Max(0, DayCap(jobPolicy[self]) - dailyAward)) ||
                balance := balance + Min(target, Max(0, DayCap(jobPolicy[self]) - dailyAward)) ||
                dailyAward := dailyAward + Min(target, Max(0, DayCap(jobPolicy[self]) - dailyAward)) ||
                decision[self] := "accepted" ||
                jobState[self] := "completed";
            end with;
        or
            decision[self] := "rejected" ||
            jobState[self] := "completed";
        or
            if attempts[self] >= MaxAttempts then
                jobState[self] := "failed";
            else
                jobState[self] := "retry";
                goto Claim;
            end if;
        end either;
    end if;
end process;

fair process PolicyAdmin = "policy-admin"
begin
RequestActivation:
    activationRequested := TRUE;
Activate:
    await activePolicy = "legacy" /\
          (\A s \in Sources : jobState[s] \notin OpenStates) /\
          dailyAward <= V2DailyCap;
    activePolicy := "v2" ||
    activationRequested := FALSE;
end process;

fair process DayBoundary = "utc-day"
begin
WaitForNeededRollover:
    while TRUE do
        await activationRequested /\ dailyAward > V2DailyCap;
        dailyAward := 0;
    end while;
end process;

fair process MobileUI = "mobile-ui"
begin
UIAction:
    while TRUE do
        if requestedRoute = "none" then
            either
                surface := "avatar";
            or
                surface := "hamburger";
            or
                surface := "none";
            or
                if surface = "hamburger" then
                    with destination \in {"profile", "credits", "rewards"} do
                        requestedRoute := destination;
                    end with;
                end if;
            end either;
        else
UICommit:
            route := requestedRoute ||
            requestedRoute := "none" ||
            surface := "none";
        end if;
    end while;
end process;
end algorithm; *)

\* BEGIN TRANSLATION
VARIABLES activePolicy, activationRequested, sourceState, jobState, jobPolicy, 
          completionKnown, completionLookups, attempts, decision, ledger, 
          balance, dailyAward, surface, route, requestedRoute, pc

vars == << activePolicy, activationRequested, sourceState, jobState, 
           jobPolicy, completionKnown, completionLookups, attempts, decision, 
           ledger, balance, dailyAward, surface, route, requestedRoute, pc >>

ProcSet == (Sources) \cup {"policy-admin"} \cup {"utc-day"} \cup {"mobile-ui"}

Init == (* Global variables *)
        /\ activePolicy = "legacy"
        /\ activationRequested = FALSE
        /\ sourceState = [s \in Sources |-> "absent"]
        /\ jobState = [s \in Sources |-> "none"]
        /\ jobPolicy = [s \in Sources |-> "none"]
        /\ completionKnown = [s \in Sources |-> FALSE]
        /\ completionLookups = [s \in Sources |-> 0]
        /\ attempts = [s \in Sources |-> 0]
        /\ decision = [s \in Sources |-> "unknown"]
        /\ ledger = [s \in Sources |-> 0]
        /\ balance = 0
        /\ dailyAward = 0
        /\ surface = "none"
        /\ route = "home"
        /\ requestedRoute = "none"
        /\ pc = [self \in ProcSet |-> CASE self \in Sources -> "SourceComplete"
                                        [] self = "policy-admin" -> "RequestActivation"
                                        [] self = "utc-day" -> "WaitForNeededRollover"
                                        [] self = "mobile-ui" -> "UIAction"]

SourceComplete(self) == /\ pc[self] = "SourceComplete"
                        /\ sourceState' = [sourceState EXCEPT ![self] = "complete"]
                        /\ pc' = [pc EXCEPT ![self] = "Enqueue"]
                        /\ UNCHANGED << activePolicy, activationRequested, 
                                        jobState, jobPolicy, completionKnown, 
                                        completionLookups, attempts, decision, 
                                        ledger, balance, dailyAward, surface, 
                                        route, requestedRoute >>

Enqueue(self) == /\ pc[self] = "Enqueue"
                 /\ /\ jobPolicy' = [jobPolicy EXCEPT ![self] = activePolicy]
                    /\ jobState' = [jobState EXCEPT ![self] = "pending"]
                 /\ pc' = [pc EXCEPT ![self] = "RecoverCompletion"]
                 /\ UNCHANGED << activePolicy, activationRequested, 
                                 sourceState, completionKnown, 
                                 completionLookups, attempts, decision, ledger, 
                                 balance, dailyAward, surface, route, 
                                 requestedRoute >>

RecoverCompletion(self) == /\ pc[self] = "RecoverCompletion"
                           /\ IF ~completionKnown[self] /\ jobState[self] \notin TerminalStates
                                 THEN /\ \/ /\ completionKnown' = [completionKnown EXCEPT ![self] = TRUE]
                                            /\ UNCHANGED <<jobState, completionLookups, decision>>
                                         \/ /\ completionLookups' = [completionLookups EXCEPT ![self] = completionLookups[self] + 1]
                                            /\ IF completionLookups'[self] >= MaxCompletionLookups
                                                  THEN /\ /\ decision' = [decision EXCEPT ![self] = "unrecoverable"]
                                                          /\ jobState' = [jobState EXCEPT ![self] = "failed"]
                                                  ELSE /\ TRUE
                                                       /\ UNCHANGED << jobState, 
                                                                       decision >>
                                            /\ UNCHANGED completionKnown
                                      /\ pc' = [pc EXCEPT ![self] = "RecoverCompletion"]
                                 ELSE /\ pc' = [pc EXCEPT ![self] = "Claim"]
                                      /\ UNCHANGED << jobState, 
                                                      completionKnown, 
                                                      completionLookups, 
                                                      decision >>
                           /\ UNCHANGED << activePolicy, activationRequested, 
                                           sourceState, jobPolicy, attempts, 
                                           ledger, balance, dailyAward, 
                                           surface, route, requestedRoute >>

Claim(self) == /\ pc[self] = "Claim"
               /\ IF jobState[self] \notin TerminalStates
                     THEN /\ IF attempts[self] >= MaxAttempts
                                THEN /\ jobState' = [jobState EXCEPT ![self] = "failed"]
                                     /\ UNCHANGED attempts
                                ELSE /\ /\ attempts' = [attempts EXCEPT ![self] = attempts[self] + 1]
                                        /\ jobState' = [jobState EXCEPT ![self] = "processing"]
                     ELSE /\ TRUE
                          /\ UNCHANGED << jobState, attempts >>
               /\ pc' = [pc EXCEPT ![self] = "Judge"]
               /\ UNCHANGED << activePolicy, activationRequested, sourceState, 
                               jobPolicy, completionKnown, completionLookups, 
                               decision, ledger, balance, dailyAward, surface, 
                               route, requestedRoute >>

Judge(self) == /\ pc[self] = "Judge"
               /\ IF jobState[self] = "processing"
                     THEN /\ \/ /\ \E target \in 1..TargetCap(jobPolicy[self]):
                                     /\ balance' = balance + Min(target, Max(0, DayCap(jobPolicy[self]) - dailyAward))
                                     /\ dailyAward' = dailyAward + Min(target, Max(0, DayCap(jobPolicy[self]) - dailyAward))
                                     /\ decision' = [decision EXCEPT ![self] = "accepted"]
                                     /\ jobState' = [jobState EXCEPT ![self] = "completed"]
                                     /\ ledger' = [ledger EXCEPT ![self] = Min(target, Max(0, DayCap(jobPolicy[self]) - dailyAward))]
                                /\ pc' = [pc EXCEPT ![self] = "Done"]
                             \/ /\ /\ decision' = [decision EXCEPT ![self] = "rejected"]
                                   /\ jobState' = [jobState EXCEPT ![self] = "completed"]
                                /\ pc' = [pc EXCEPT ![self] = "Done"]
                                /\ UNCHANGED <<ledger, balance, dailyAward>>
                             \/ /\ IF attempts[self] >= MaxAttempts
                                      THEN /\ jobState' = [jobState EXCEPT ![self] = "failed"]
                                           /\ pc' = [pc EXCEPT ![self] = "Done"]
                                      ELSE /\ jobState' = [jobState EXCEPT ![self] = "retry"]
                                           /\ pc' = [pc EXCEPT ![self] = "Claim"]
                                /\ UNCHANGED <<decision, ledger, balance, dailyAward>>
                     ELSE /\ pc' = [pc EXCEPT ![self] = "Done"]
                          /\ UNCHANGED << jobState, decision, ledger, balance, 
                                          dailyAward >>
               /\ UNCHANGED << activePolicy, activationRequested, sourceState, 
                               jobPolicy, completionKnown, completionLookups, 
                               attempts, surface, route, requestedRoute >>

SourceActor(self) == SourceComplete(self) \/ Enqueue(self)
                        \/ RecoverCompletion(self) \/ Claim(self)
                        \/ Judge(self)

RequestActivation == /\ pc["policy-admin"] = "RequestActivation"
                     /\ activationRequested' = TRUE
                     /\ pc' = [pc EXCEPT !["policy-admin"] = "Activate"]
                     /\ UNCHANGED << activePolicy, sourceState, jobState, 
                                     jobPolicy, completionKnown, 
                                     completionLookups, attempts, decision, 
                                     ledger, balance, dailyAward, surface, 
                                     route, requestedRoute >>

Activate == /\ pc["policy-admin"] = "Activate"
            /\ activePolicy = "legacy" /\
               (\A s \in Sources : jobState[s] \notin OpenStates) /\
               dailyAward <= V2DailyCap
            /\ /\ activationRequested' = FALSE
               /\ activePolicy' = "v2"
            /\ pc' = [pc EXCEPT !["policy-admin"] = "Done"]
            /\ UNCHANGED << sourceState, jobState, jobPolicy, completionKnown, 
                            completionLookups, attempts, decision, ledger, 
                            balance, dailyAward, surface, route, 
                            requestedRoute >>

PolicyAdmin == RequestActivation \/ Activate

WaitForNeededRollover == /\ pc["utc-day"] = "WaitForNeededRollover"
                         /\ activationRequested /\ dailyAward > V2DailyCap
                         /\ dailyAward' = 0
                         /\ pc' = [pc EXCEPT !["utc-day"] = "WaitForNeededRollover"]
                         /\ UNCHANGED << activePolicy, activationRequested, 
                                         sourceState, jobState, jobPolicy, 
                                         completionKnown, completionLookups, 
                                         attempts, decision, ledger, balance, 
                                         surface, route, requestedRoute >>

DayBoundary == WaitForNeededRollover

UIAction == /\ pc["mobile-ui"] = "UIAction"
            /\ IF requestedRoute = "none"
                  THEN /\ \/ /\ surface' = "avatar"
                             /\ UNCHANGED requestedRoute
                          \/ /\ surface' = "hamburger"
                             /\ UNCHANGED requestedRoute
                          \/ /\ surface' = "none"
                             /\ UNCHANGED requestedRoute
                          \/ /\ IF surface = "hamburger"
                                   THEN /\ \E destination \in {"profile", "credits", "rewards"}:
                                             requestedRoute' = destination
                                   ELSE /\ TRUE
                                        /\ UNCHANGED requestedRoute
                             /\ UNCHANGED surface
                       /\ pc' = [pc EXCEPT !["mobile-ui"] = "UIAction"]
                  ELSE /\ pc' = [pc EXCEPT !["mobile-ui"] = "UICommit"]
                       /\ UNCHANGED << surface, requestedRoute >>
            /\ UNCHANGED << activePolicy, activationRequested, sourceState, 
                            jobState, jobPolicy, completionKnown, 
                            completionLookups, attempts, decision, ledger, 
                            balance, dailyAward, route >>

UICommit == /\ pc["mobile-ui"] = "UICommit"
            /\ /\ requestedRoute' = "none"
               /\ route' = requestedRoute
               /\ surface' = "none"
            /\ pc' = [pc EXCEPT !["mobile-ui"] = "UIAction"]
            /\ UNCHANGED << activePolicy, activationRequested, sourceState, 
                            jobState, jobPolicy, completionKnown, 
                            completionLookups, attempts, decision, ledger, 
                            balance, dailyAward >>

MobileUI == UIAction \/ UICommit

Next == PolicyAdmin \/ DayBoundary \/ MobileUI
           \/ (\E self \in Sources: SourceActor(self))

Spec == /\ Init /\ [][Next]_vars
        /\ WF_vars(Next)
        /\ \A self \in Sources : WF_vars(SourceActor(self))
        /\ WF_vars(PolicyAdmin)
        /\ WF_vars(DayBoundary)
        /\ WF_vars(MobileUI)

\* END TRANSLATION

OpenJobExists == \E s \in Sources : jobState[s] \in OpenStates

TypeOK ==
    /\ activePolicy \in Policies
    /\ activationRequested \in BOOLEAN
    /\ sourceState \in [Sources -> {"absent", "complete"}]
    /\ jobState \in [Sources -> JobStates]
    /\ jobPolicy \in [Sources -> (Policies \cup {"none"})]
    /\ completionKnown \in [Sources -> BOOLEAN]
    /\ completionLookups \in [Sources -> 0..MaxCompletionLookups]
    /\ attempts \in [Sources -> 0..MaxAttempts]
    /\ decision \in [Sources -> Decisions]
    /\ ledger \in [Sources -> 0..LegacyDailyCap]
    /\ balance \in 0..(LegacyDailyCap * Cardinality(Sources))
    /\ dailyAward \in 0..LegacyDailyCap
    /\ surface \in Surfaces
    /\ route \in Routes
    /\ requestedRoute \in (Routes \cup {"none"})

JobHasCoherentSnapshot ==
    \A s \in Sources : jobState[s] # "none" => jobPolicy[s] \in Policies

PolicySnapshotPinned ==
    \A s \in Sources :
        jobPolicy[s] = "legacy" \/ jobPolicy[s] = "v2" \/ jobState[s] = "none"

V2TargetsAreStricter ==
    /\ V2TargetCap <= LegacyTargetCap
    /\ V2DailyCap <= LegacyDailyCap

LedgerTotal ==
    IF Sources = {OldSource}
    THEN ledger[OldSource]
    ELSE ledger[OldSource] + ledger[NewSource]

LedgerMatchesBalance == balance = LedgerTotal

NoAwardWithoutAcceptance ==
    \A s \in Sources : ledger[s] > 0 => decision[s] = "accepted"

TerminalMissingCompletionIsVisible ==
    \A s \in Sources :
        completionLookups[s] = MaxCompletionLookups /\ ~completionKnown[s]
        => jobState[s] = "failed" /\ decision[s] = "unrecoverable"

V2AwardCapRespected ==
    \A s \in Sources : jobPolicy[s] = "v2" => ledger[s] <= V2TargetCap

ActivationGuardPreserved ==
    activePolicy = "v2" => dailyAward <= V2DailyCap

AvatarIsIdentityOnly == surface = "avatar" => requestedRoute = "none"

MenuShapeCorrect ==
    /\ StudentMenu[2] = "Learning Credits"
    /\ StudentMenu[3] = "Rewards"
    /\ TutorMenu[2] = "Learning Credits"
    /\ TutorMenu[3] = "Rewards"

AwardFormulaBoundaries ==
    /\ StrokeAward(1) = 1
    /\ StrokeAward(250) = 1
    /\ StrokeAward(251) = 2
    /\ StrokeAward(1250) = 5
    /\ StrokeAward(5000) = 5
    /\ PhotoAward(1) = 1
    /\ PhotoAward(10) = 10
    /\ PhotoAward(11) = 10

TierBoundaries ==
    /\ TierOf(0) = "Seed"
    /\ TierOf(99) = "Seed"
    /\ TierOf(100) = "Scribe"
    /\ TierOf(499) = "Scribe"
    /\ TierOf(500) = "Pathfinder"
    /\ TierOf(1499) = "Pathfinder"
    /\ TierOf(1500) = "Beacon"
    /\ TierOf(3999) = "Beacon"
    /\ TierOf(4000) = "Luminary"

AllJobsEventuallyTerminal ==
    \A s \in Sources : jobState[s] = "pending" ~> jobState[s] \in TerminalStates

ActivationEventuallySucceeds == activationRequested ~> activePolicy = "v2"

RequestedNavigationCommits ==
    \A destination \in {"profile", "credits", "rewards"} :
        requestedRoute = destination ~> route = destination

=================================================================
