# Performanceplan für Pascal SM61

Sol soll Padra auf der GTX 1070 Max-Q in beiden langen Benchmarkprofilen mindestens auf die Geschwindigkeit des vorhandenen llama.cpp-Setups bringen. Dafür sind zwei Arbeiten entscheidend: den ungünstigen FP16-Codec im Expertenpfad ersetzen und einen echten, nach Layern und Experten gebündelten Prefill implementieren. Decode braucht zusätzlich schnellere Attention und eine gezielte Optimierung der verbleibenden Projektionen.

Die Zielgeschwindigkeit ist ein Abnahmekriterium, keine bereits belegte Prognose. Ein isoliertes GPU-Experiment zeigt allerdings einen konkreten ersten Gewinn: Die unveränderte Expertenprojektion wird mit einem bitgleichen Codec ungefähr 3,5–4-mal schneller. Dieser Faktor gilt für den getesteten Kernel, nicht für die gesamte Inferenz.

Stand: 13. Oktober 2026. Codebasis: `codex/pascal-sm61-port`, `1cadc65`; der gemessene Runtime-Code stammt aus `b3ca187`. Ausgangsmessungen und Identitäten stehen in [Pascal-Validierung](pascal-sm61-validation.md). Die folgenden Aufgaben ändern zunächst weder das native EXL3-K4-Artefakt noch BF16-KV. Alternative Arithmetik bekommt ein eigenes, geprüftes Profil.

## Ziel und Messvertrag

| Profil | Padra bisher | llama.cpp bisher | Mindestziel Padra | Weiteres Optimierungsziel |
| --- | --- | --- | --- | --- |
| Prefill 2000 | 111,410 s / 17,95 Tok/s | 2,245 s / 890,99 Tok/s | höchstens 2,245 s | mindestens 1000 Tok/s |
| Prefill 4000 | 230,814 s / 17,33 Tok/s | 4,476 s / 893,64 Tok/s | höchstens 4,476 s | mindestens 1000 Tok/s |
| Decode nach 2000 | 61,458 s / 16,26 Tok/s | 17,473 s / 57,17 Tok/s | höchstens 17,473 s | mindestens 65 Tok/s |
| Decode nach 4000 | 64,229 s / 15,55 Tok/s | 17,906 s / 55,79 Tok/s | höchstens 17,906 s | mindestens 65 Tok/s |

Jedes Profil erzeugt tatsächlich 1000 Ausgabetokens; Decode zählt 999 Forward-Schritte. Prefill enthält die erste Tokenauswahl. Kontext 8192, identische Eingabetokens, Greedy, kein Promptcache, keine Spekulation, ein Request gleichzeitig. Gewichteladen, Tokenisierung und Warteschlange bleiben außerhalb der Phasenzeit. Ein schnellerer kurzer `tg128`-Test ersetzt keine dieser vier Abnahmen.

Die vorhandenen Zahlen sind jeweils einzelne warme Läufe. Vor der endgültigen Abnahme beide Engines erneut mit identischen gespeicherten Requests messen: je mindestens fünf warme Wiederholungen, wechselnde Reihenfolge der Engines, keine gleichzeitige GPU-Nutzung. Median, Streuung und gepaarte Laufzeitverhältnisse ausweisen. Erreicht ist Parität, wenn Padra in **allen vier Phasen** mindestens den zeitnah gemessenen llama.cpp-Median erreicht; ein Bootstrap-Konfidenzintervall des gepaarten Speedups soll nicht unter 1 fallen. Bei unklarer Streuung weitere Wiederholungen. Zusätzlich die historischen Ziele oben berichten, damit eine verlangsamte Referenz keinen Erfolg vortäuscht.

GPU-Takt, Speichertakt, Temperatur, Leistung und Drosselungsgründe protokollieren. Keine Takt-/Power-Änderung zwischen den Engines. Profilierte Laufzeiten nicht als finale Geschwindigkeit verwenden. HTTP-Gesamtzeit und TTFT zusätzlich festhalten, damit Arbeit nicht nur aus der Prefill-Zeit in Request-Vorbereitung verschoben wird.

EXL3 K4/BF16 und Q4_K_M/Q8_0 sind unterschiedliche Rechenprofile; auch die Quellrevisionen der Artefakte unterscheiden sich. Das Ziel ist praktische Setup-Parität. Identische Tokenisierung allein belegt keine identische Modellqualität.

## Befunde im aktuellen Code

| Befund | Code auf der Ausgangsbasis | Konsequenz |
| --- | --- | --- |
| Pro Prompttoken ein vollständiger Forward-Graph | `src/cuda/inference_simt.cu:184`, innere Schleife ab 187 | `prefill_chunk=256` bündelt keine Matrixarbeit. Nur Chunk-Grenzen, Abbruch und Commit werden gebündelt. |
| 18 Kernelknoten je Layer, 28 Layer, plus Embedding | `src/cuda/inference_simt.cu:129` | 505 Forward-Knoten je Token; bei 4000 Tokens etwa 2,02 Millionen Knoten. CUDA Graphs existieren bereits. Ein weiterer Graph um dieselbe Token-Schleife löst die fehlende Wiederverwendung nicht. |
| FP16-FMA im inneren EXL3-Codec | `src/cuda/expert_exl3_selected.cuh:16` | `__hfma2` wird im SM61-Binary tatsächlich zu `HFMA2`. Dieser Chip hat sehr niedrigen FP16-Arithmetikdurchsatz. |
| Separate Transformation für Gate, Up und Down | `src/cuda/expert_exl3_selected.cuh:52`, `expert_exl3.cuh:21` | `InputHad` verwendet für H128 wiederholt blockweite Barrieren. Gate und Up haben eigene `suh` und dürfen ihre transformierten Eingaben nicht blind teilen. |
| FP8 als Speicherformat, FP32 als Rechenformat | `src/cuda/fp8_projection_simt.cuh:5`, `fp8_vector.cuh:3` | Jedes GEMV dekodiert FP8-Gewichte und -Eingaben wiederholt. Es gibt hier keine native FP8-Matrixbeschleunigung. |
| Nur 32 Attention-CTAs pro Layer | `src/cuda/simt_attention.cuh:20` | Ein CTA pro Q-Kopf; die acht Q-Köpfe eines KV-Kopfs laden dessen K/V unabhängig. Cache kann DRAM-Verkehr reduzieren, aber die redundanten Ladeinstruktionen bleiben. |
| Lange serielle PV-Schleife | `src/cuda/simt_attention.cuh:48` | Je Ausgabedimension läuft ein Thread über sämtliche sichtbaren Positionen; nur 128 der 256 Threads berechnen PV. |
| FP64-Exponentialfunktion im laufenden Softmax | `src/cuda/scalar_casts.cuh:6`, `simt_attention.cuh:44` | `ExpRN` nutzt `exp(double)`. Das ist ein anderer Engpass als die bereits vorhandene SiLU-Tabelle, deren FP64-Aufbau nur einmal erfolgt. |
| BF16-Head mit 98.304 × 2304 Gewichten | `src/cuda/head_bf16_prefetch.cuh:5` | Rund 453 MB Gewichte pro generiertem Token; kompensierte Akkumulation. Im Prefill wird der Head bereits nur am Ende gerechnet. |
| BF16-KV und feste Speicheraufnahme | `src/runtime/simt_memory.h:6`, `inference_simt.cu:36` | Erweiterungen müssen in die echte 8-GiB-Aufnahme inklusive 700 MiB Reserve und 42 MiB Snapshot passen. |

Der Blackwell-Pfad ist eine andere Implementierung: `src/runtime/model_runtime.cpp:10` aktiviert gruppierten Prefill, MMA-Projektionen, andere Attention-Kernel und fusioniertes Decode. Seine Performance lässt sich durch diese Dispatch-Auswahl erklären, aber ohne Blackwell-Messung nicht zahlenmäßig auf Pascal übertragen. `cmake/PadraCuda.cmake` trennt die CUDA-Übersetzungseinheiten bereits sinnvoll.

NVIDIA dokumentiert für GP104 einen FP16-Durchsatz von nur 1/64 des FP32-Durchsatzes, dagegen schnelle Integer-Dotprodukte. FP16-Speicherung und FP16-Rechnen sind daher getrennt zu behandeln. SM61 hat außerdem keine Tensor Cores; pro CTA stehen höchstens 48 KiB Shared Memory zur Verfügung. [NVIDIA Pascal Tuning Guide](https://docs.nvidia.com/cuda/archive/12.4.1/pascal-tuning-guide/index.html)

Die lokal geprüfte llama.cpp-Revision `77757a34672a888845a09ab45728b8d34aa475ff` schließt SM61 explizit aus `FAST_FP16_AVAILABLE` aus (`ggml/src/ggml-cuda/common.cuh:271`). Ihre MMQ-Auswahl unterstützt quantisierte Matrixmultiplikation mit DP4A auf dieser Architektur (`mmq.cu:356`, `mmq.cuh`, `vecdotq.cuh`). Das liefert konkrete Implementierungsreferenzen; welche Kernel im Vergleichslauf tatsächlich dominieren, muss ein Trace zeigen. EXL3 ist kein lineares Q4_K-Format und kann diese Dotprodukte nicht unverändert übernehmen.

## Gemessenes Profil des unveränderten Servers

CUDA 12.4 `nvprof` konnte den bestehenden SM61-Server mit dem echten, unveränderten Checkpoint profilieren. Zwei Requests mit jeweils 24 Prompt- und 32 Ausgabetokens ergeben 110 vollständige Forward-Schritte und 64 Head-Aufrufe. Die Übersicht umfasst beide Requests, einschließlich des ersten Laufs. Sie ist eine kurze Bottleneck-Messung und kein Ersatz für die langen Abnahmeprofile.

| Operatorgruppe | Anteil an der aufsummierten GPU-Kernelzeit | Abgeleitete Kernelzeit |
| --- | ---: | ---: |
| EXL3-Projektionen, Gate/Up/Down | 70,3 % | 37,25 ms pro Forward über 28 Layer |
| QKV-Projektionen | 10,5 % | 5,58 ms pro Forward |
| O-Projektionen | 8,7 % | 4,61 ms pro Forward |
| BF16-Head | 2,9 % | 2,60 ms pro Head-Aufruf |
| Attention, lokal und global zusammen | 0,8 % | 0,42 ms pro Forward bei diesen kurzen Kontexten |
| Übrige Kernel | 6,8 % | Norm, Quantisierung, Routing, H128 und Steuerung |

Bezugsgröße sind rund 5,830 s summierte Kernelzeit. Insbesondere der einmalige Gewichtsupload von rund 0,682 s ist aus diesen Anteilen ausgeschlossen. CPU-Zeit, Graph-Lücken und Transfers sind ebenfalls kein Bestandteil der Prozentbasis. Die Head-Anzahl unterscheidet sich von der Forward-Anzahl; deshalb nicht alle Zeilen als identischen Pro-Token-Anteil interpretieren.

Die Messung macht den EXL3-Codec zur ersten Priorität und QKV/O zum nächsten belegten Rechenposten. Ein hypothetischer Faktor 3,7 ausschließlich für die Expertenprojektion würde 37,25 ms auf etwa 10,1 ms reduzieren. QKV/O und Head benötigen schon weitere 12,8 ms, noch ohne Attention, übrige Kernel und Lücken. **Der Codec allein reicht daher nicht für die angestrebten etwa 17,5 ms pro Decode-Schritt.** Langkontext-Attention kann die verbleibende Lücke weiter vergrößern und muss separat profiliert werden.

Rohdaten: `nvprof-short.csv`, `nvprof-request-0.json`, `nvprof-request-1.json` und `profile-summary.json` im Experimentverzeichnis. `profile_nvprof.py` enthält den reproduzierbaren Serveraufruf; es nutzt den vorhandenen Build unter `/tmp/padra-sm61-build`. Der Server wurde anschließend beendet.

## Bereits geprüftes Codec-Experiment

Der Test liegt unter `/srv/agent/workspaces/padra-pascal-plan-20261013/`. Er benutzt eine Kopie von `exl3_selected::Projection` und synthetische, voneinander getrennte Gewichte für acht Experten. Das Modellartefakt und der Produktionscode bleiben unverändert.

Der Ersatz berechnet den Hash-Byte-Summenwert `s` und anschließend:

```cpp
float value = __fmaf_rn(float(1024 + s), half_constant_a, half_constant_b);
float decoded = __half2float(__float2half_rn(value));
```

Die Konstanten sind exakt die als FP32 dargestellten FP16-Bitmuster `0x1eee` und `0xc931`. Die FP16-Rundung bleibt erhalten. `0x6400+s` als Half entspricht in diesem Wertebereich `1024+s`. Die zweite Variante ersetzt zusätzlich die vier Byte-Additionen durch das **unsigned** `__dp4a(hash, 0x01010101U, 0U)`. Das beschleunigt nur die exakte Hash-Summe und quantisiert keine Aktivierungen.

| Projektion für acht Experten | Original | Nur FP32-Codec | FP32-Codec plus DP4A | Faktor der kombinierten Variante |
| --- | ---: | ---: | ---: | ---: |
| Gate/Up, 896 × 2304 | 0,447 ms | 0,142 ms | 0,126 ms | 3,55 |
| Down, 2304 × 896 | 0,449 ms | 0,136 ms | 0,113 ms | 3,97 |

Werte: Median aus fünf CUDA-Event-Messblöcken mit jeweils 100 Kernelaufrufen; Aufwärmphase, abwechselnde Reihenfolge von Original und kombinierter Variante. Das ist ein kleiner, wiederholt verwendeter Working Set und kein vollständiger Modelllauf. Die Unterschiede zu realen Gewichts-/Cachezugriffen und Langzeitthermik begrenzen die Übertragbarkeit.

Geprüft wurden alle 65.536 Codec-Zustände gegen den ursprünglichen GPU-Codec sowie bitgleiche vollständige Projektionsausgaben für beide Produktionsformen. Die disassemblierte ursprüngliche Projektion enthält `HFMA2`; die kombinierte Variante enthält keine `HFMA2`, sondern FP32-FMA und Integer-Dot-Instruktionen. Die vorhandene FP16-Rundungssemantik bleibt trotz dieser Instruktionsänderung erhalten. Ein Checkpoint-Qualitätslauf mit eingebautem Ersatz steht noch aus.

Reproduktion mit CUDA 12.4 und GCC 13:

```sh
nvcc -ccbin g++-13 -std=c++20 -O3 -lineinfo -arch=sm_61 \
  -I/home/sebastian/padra/include -I/home/sebastian/padra/src \
  -I/home/sebastian/padra/src/cuda \
  /srv/agent/workspaces/padra-pascal-plan-20261013/probe.cu \
  -o /srv/agent/workspaces/padra-pascal-plan-20261013/probe
/srv/agent/workspaces/padra-pascal-plan-20261013/probe
```

`candidate.cuh`, `candidate_scalar.cuh`, `probe.cu` und `codec-results-warm.txt` enthalten die Kandidaten und Rohdaten. Der Probe-Code bindet die Testhilfen aus dieser Codebasis ein. Für einen späteren Checkout zuerst die Ausgangsrevision herstellen oder die Quellen passend archivieren.

## Umsetzung in einzelnen Arbeitspaketen

Die Reihenfolge ist P0 → P1 → P2/P3/P4 → P5 → P6 → P7. P2 bis P4 ergeben gemeinsam den produktiven Prefill. Die QKV/O-Arbeit aus P6 bereits mit P3 messen, weil der kurze Trace diese Gruppe als zweitgrößten Posten belegt. P5 kann nach P1 vorgezogen werden, falls das Langkontext-Decode-Profil Attention als dominierenden Rest zeigt. Nach jedem Paket einen eigenständigen Commit mit Messungen erstellen. Keine pauschalen Speedup-Faktoren verschiedener Pakete miteinander multiplizieren.

### P0 Messung und numerische Referenz

**Lieferung:** ein SM61-Operatorbenchmark, ein kurzer Laufzeittrace und wiederholbare Referenzdaten für echte Modellaktivierungen. Aufzuschlüsseln sind EXL3-Codec/Projektion, H128, QKV/O, lokale/globale Attention, Router/Norm, Head und CPU/GPU-Lücken. Prefill und Decode getrennt messen, Layer-Typ und absolute Kontextposition angeben.

1. Vor Änderungen die langen Requests wiederholen und Kernelzeiten bei Kontext 128, 1024, 2000, 4000 und 8000 erfassen. Bei 8000 nur so viele neue Tokens erzeugen, wie in 8192 passen. Einen nicht profilierten Kontrolllauf danebenstellen.
2. Einen Diagnosemodus außerhalb des Server-Hotpaths vorsehen: ausgewählte echte Layer-Eingaben, Top-8-IDs, Routinggewichte, Q/K/V und Logits speichern. Teacher-forced Replay derselben Tokens ermöglichen; aktuell lehnt `ValidateSimtOptions` die entsprechenden SM120-Diagnoseoptionen ab. Neue SM61-Diagnostik explizit implementieren.
3. CUDA Events um Operatorgruppen im Diagnosepfad setzen; Zeitmessung nicht durch Synchronisation nach jedem Produktionskernel verfälschen. Eager-Diagnose und Graph-Replay vergleichen.
4. Mit CUDA 12.x, `-lineinfo` und `ptxas`-Ressourcenbericht bauen. SASS auf `HFMA2`, FP64-Instruktionen, Registerverbrauch und Spills prüfen. Profiler-Konfiguration mit abspeichern.

**Profilerwahl:** Nsight Compute unterstützt Pascal nicht. CUDA 12.4 `nvprof` oder CUDA-Event-Messungen verwenden; Nsight Systems eignet sich mit kompatibler Installation für die Timeline. Auf dieser Maschine konnte Nsight Systems 2023.4 einen QDSTRM-Trace aufnehmen, aber mangels Importer nicht in einen auswertbaren Report umwandeln. Der anschließende `nvprof`-Lauf lieferte die oben ausgewerteten Kernelzeiten. Hardware-Counter-Zugriff wurde damit noch nicht geprüft; bei fehlenden Rechten mit Events und SASS weiterarbeiten. [NVIDIA Profiler Guide](https://docs.nvidia.com/cuda/archive/12.4.1/profiler-users-guide/index.html), [Nsight Compute GPU-Support](https://docs.nvidia.com/nsight-compute/ReleaseNotes/)

**Abnahme:** Die zeitlich erklärten Operatorgruppen und gemessenen Lücken decken die Phasenlaufzeit weitgehend ab; Restabweichung und Profiling-Overhead werden ausgewiesen. Ein Amdahl-Budget nennt, wie viel Zeit jede geplante Änderung tatsächlich einsparen kann.

### P1 Exakten Pascal-Codec integrieren

**Dateien:** `expert_exl3_selected.cuh`, gegebenenfalls ein eigenes `expert_exl3_pascal_codec.cuh`, `tests/simt_kernels.cu`.

1. Den gemessenen FP32-Codec mit FP16-Rundung hinter einem eindeutigen SM61-Dispatch integrieren. Die Blackwell-Codec-/MMA-Pfade behalten ihre Arithmetik.
2. FP32-Ersatz und unsigned DP4A-Hash zunächst separat messen, dann die schnellere geprüfte Kombination wählen. Die vorhandene Datei `expert_exl3_dot_codec.cuh` liefert den Hash-Mechanismus, enthält aber weiterhin FP16-FMA; sie allein beseitigt den SM61-Engpass nicht.
3. Alle 65.536 Zustände exakt prüfen. Projektionstests auf reale Dimensionen, unterschiedliche Experten, Sidecars, Gate/Up/Down, Vorzeichen und Grenzwerte erweitern. Den aktuellen 128×128-Test mit identischen Experten ergänzen.
4. Reale Checkpoint-Smokes, teacher-forced Logits und beide Langbenchmarks ausführen. Ausschließlich Codec-Arithmetik geändert: Bitgleichheit gegen die SIMT-Basis ist das Ziel.

**Abnahme:** keine `HFMA2` im SM61-Hotloop, unveränderte Checkpoint-Identität, bitgleiche Referenzwerte, klarer Kernel- und End-to-End-Gewinn. Die 3,5–4× aus dem Mikrobenchmark sind ein Prüfwert, keine End-to-End-Vorgabe. Danach neu profilieren.

### P2 Echten Prefill nach Layern ausführen

**Dateien:** `inference_simt.cu`, neues `simt_prefill.cuh`, `simt_memory.h`, `prefill_state.h`, `chunk_graph_contract.h`, Runtime-Profilangaben und `tests/checkpoint_smoke.py`.

Den Ablauf von „Token → 28 Layer“ auf „Chunk → 28 Layer → alle aktiven Zeilen“ umstellen:

```text
Prompt-IDs einmal auf die GPU
für jeden Chunk und darin jeden Layer
  Norm und QKV aller Zeilen
  QK-Norm und RoPE mit absoluten Positionen
  kausale Attention gegen alten Cache und den aktuellen Chunk
  KV des Layers nach den nötigen Reads committen
  O-Projektion und Residual/Postnorm aller Zeilen
  Top-8-Routing je Zeile und Gruppierung nach Experte
  Gate/Up, SiLU, Down und geordnetes Combine
Head und Auswahl nur für die letzte Promptzeile
```

`inference.cu:834` enthält das vorhandene `prefillChunk` als Ablaufreferenz. Nur kompatible Scheduler-/Kontrollbausteine übernehmen; seine MMA-, `cp.async`- und SM120-Kernel sind keine Pascal-Implementierung.

Mit physischem Batch 256 beginnen. Für Tails aktive Zeilenzahl im Device-State abbilden; keine Ausführung ungültiger Padding-Zeilen und keine Weiterverwendung alter Routingdaten. Interne Tile-Größen 16/32/64 sind unabhängig von der öffentlichen Chunk-Auswahl 256/512/2048. Ein neuer öffentlicher Chunkwert erfordert Änderungen an CLI, Validierung und Speicherplan.

**Kritischer Cachevertrag:** Ein vorzeitig geschriebener Chunk kann im lokalen 1024er-Ring KV zerstören, das frühere Queries des Chunks noch benötigen. Alten Ring und neue Chunk-KV getrennt lesen; erst danach committen. Für Zeile `t` gilt die absolute Queryposition `base+t`, kausal bis einschließlich dieser Position und lokal ab `max(0, base+t-1023)`. Einen Snapshot-Anker innerhalb eines Chunks zunächst durch Aufteilen an diesem Anker behandeln; erst später eine bewiesene Snapshot-Synthese optimieren.

Graphs mit stabilen Adressen und Device-State aufnehmen. Keine Capture-/Allokationsarbeit je Request. Voller Chunk und Tail müssen denselben mathematischen Ablauf haben. Der bisherige Smoke-Test erwartet genau zwei Captures für SIMT; diese Erwartung auf den neuen, begrenzten Capture-Vertrag umstellen. Fehlerzustand, Abbruch, `append_base`, Pending-Token und `CommitChunk` bleiben korrekt.

**Abnahme:** Sequenzieller und gebündelter Referenzpfad stimmen für die vereinbarte Arithmetik überein; Ring-Wrap, Restore und Chunk-Tails funktionieren. Der Trace zeigt Matrixarbeit über mehrere Tokens, nicht bloß weniger Hostaufrufe. P2 allein darf noch langsam sein; es ist das Fundament für P3/P4.

### P3 QKV und O als echte Batch-Matrizen rechnen

**Dateien:** `fp8_projection_simt.cuh`, neue Pascal-Batch-Kernel, optional CMake-Link gegen `CUDA::cublas` als Vergleichskandidat.

Die FP8-Werte pro Tile einmal in FP32 dekodieren und über mehrere Tokenzeilen wiederverwenden. Auch die Eingabedekodierung je Zeile wiederverwenden. Zeilenskalen und die vorhandenen BF16-Ausgabegrenzen erhalten. Bei Einzeltoken-Decode separat messen, ob vorab dekodierte Eingaben im Shared Memory die wiederholte Konvertierung einsparen.

Zwei Prefill-Kandidaten vergleichen:

- Eigener FP32-SIMT-Tile-Kernel mit N/M/K-Wiederverwendung. Zunächst die bisherige Akkumulationsreihenfolge pro Ausgabe beibehalten, sofern wirtschaftlich; Tiles anhand Register-/Shared-Memory-Bedarf wählen.
- Begrenzte Dekodierung einer Matrix oder eines Layers nach FP32 plus SGEMM als Leistungsreferenz. Dekodierzeit, Scratch, Library-Workspace und Graph-Verträglichkeit in die Messung einbeziehen. SGEMM ändert gegebenenfalls die Reduktionsreihenfolge und braucht dann den numerischen Gate aus P7.

Keine vollständige FP32-Kopie sämtlicher Projektionsgewichte resident halten. Die FP8-Payload dieser Projektionen umfasst schon rund 595 MB; zusätzliches FP32 aller Layer wären rund 2,38 GB. Ein Layer umfasst dagegen etwa 21,2 MB FP8 beziehungsweise 84,9 MB FP32. FP16-GEMM auf GP104 ist kein sinnvoller Standardkandidat; ein anderer Datentypname allein schafft keine Beschleunigung.

**Abnahme:** Zeit pro vollständigem Layerblock für M=1/16/32/64/128/256 und reale Matrizen ausweisen. Ein Schritt zu höherem Durchsatz gilt nur einschließlich Dekodierung und Skalierung. Decode darf durch Prefill-Workspace oder Dispatch nicht schlechter werden.

### P4 MoE nach Experten gruppieren und Gewichte wiederverwenden

**Dateien:** `expert_schedule.cuh`, `exl3_masked_schedule.h`, neue `expert_exl3_prefill_simt.cuh`, Pascal-Prefill-Dispatcher.

Bei 256 Tokens entstehen 2048 Token-Experten-Paare. Gleichmäßiges Routing ergäbe durchschnittlich 32 Zeilen pro Experte. Das ist die Gelegenheit, denselben komprimierten Gewichtsblock einmal zu dekodieren und für mehrere Zeilen zu verwenden. Die echte Verteilung messen; 32 ist keine garantierte Auslastung.

1. Stabile GPU-Gruppierung nach Experte mit Rückabbildung auf Token und Top-8-Slot. Bestehendes `fast_expert::Group` als Referenz verwenden. Kein Download von Expertenzählern und kein Host-Dispatch pro Experte/Layer.
2. `suh` und H128 für jedes Token-Experten-Paar korrekt anwenden. Gate und Up dürfen Eingabeladevorgänge bündeln, haben aber getrennte Sidecars und Transformationen. Down verarbeitet die jeweilige aktivierte Expertenzeile.
3. Tiled FP32-GEMM mit dem exakten Codec aus P1 entwickeln. Komprimierte Gewichte und dekodierte Tile-Werte über M-Zeilen wiederverwenden. M=8/16/32 und unterschiedliche N/K-Tiles prüfen; Auswahl nach tatsächlich belegten Zeilen.
4. Für sehr kleine Gruppen den optimierten GEMV-Pfad nutzen. Maskierte Tails dürfen keine Ausgaben oder Top-8-Slots verlieren. Stark schiefes Routing, leere Experten und einzelne belegte Zeilen gehören in die Tests.
5. Geordnet in den ursprünglichen Top-8-Slots kombinieren. Ungeordnete Floating-Point-Atomics würden die bisherige Reproduzierbarkeit verändern.

Als Vergleich darf ein einzelner Experte oder Tile vorübergehend dekodiert und mit SGEMM gerechnet werden. Keine vollständige Expansion des Experten-Layers: 64 Experten mit drei Matrizen benötigen schon als FP16 rund 793 MB zusätzlich. Für acht GiB ist ein kleiner, wiederverwendbarer Tile-/Expertenscratch vorgesehen.

**Abnahme:** derselbe reale Chunk erreicht hohe Wiederverwendung auch einschließlich Gruppierung, H128, Aktivierung und Scatter. Anzahl dekodierter Gewichte sinkt gegenüber `256 × GEMV` entsprechend der tatsächlichen Gruppengrößen. Rohzahlen nach Gruppenbelegung berichten. Nach P2–P4 ist Prefill ≥300 Tok/s eine diagnostische Zwischenmarke; das Endziel bleibt etwa 900 Tok/s oder besser. Wird sie klar verfehlt, den dominierenden Operator beheben, bevor mehr Varianten entstehen.

### P5 Attention für Prefill und Decode parallelisieren

**Dateien:** `simt_attention.cuh`, neue Pascal-Attention-Kernel. `attention_decode_fast.cuh` als FP32-SIMT-Split/Merge-Referenz prüfen; dessen Math-/State-Vertrag unterscheidet sich vom jetzigen Fallback.

Für Decode Kontext in Tiles aufteilen und mehr unabhängige CTAs erzeugen. KV-Ladevorgänge innerhalb einer GQA-Gruppe wiederverwenden; Gruppengrößen 2/4/8 und Kontext-Tiles etwa 64/128/256 vergleichen. Zu große Gruppen können durch Registerdruck langsamer werden. Den kleinen lokalen Kontext und den globalen Kontext getrennt tunen.

Für Prefill mehrere Queryzeilen pro Tile rechnen und K/V über diese Queries wiederverwenden. Keinen vollständigen `[Chunk, 32, Kontext]`-Scoretensor aufbauen: Schon 256×32×8192 FP32-Werte belegen 256 MiB. Ein zweiter Wahrscheinlichkeitstensor würde den verfügbaren Spielraum weiter überschreiten.

Hier sind zwei numerisch verschiedene Lösungen zu unterscheiden:

- **Bestehende Semantik:** gerundete BF16-Scores, globaler Max-/Summenbezug, `ExpRN`, gerundete BF16-Wahrscheinlichkeiten und kompensiertes PV erhalten. Tiling kann mehrere Durchläufe oder Rekonstruktion der Scores erfordern. Geänderte Reduktionsbäume sind nicht automatisch bitgleich.
- **Schnelles, separat ausgewiesenes Profil:** FP32-Online-Softmax mit Split/Merge und `expf` beziehungsweise geprüftem `__expf`. Dieser Ablauf vermeidet die serielle PV-Schleife und FP64-Exponentialfunktion, verändert aber die Rundungsgrenzen. Das ist eine Qualitätsänderung und kein rein mechanischer Port.

Den exakten Referenzpfad behalten. Erst zeitlich prüfen, wie viel FP64-Exp gegenüber PV, Kompensation und Ladeverkehr kostet. Kein globales `--use_fast_math` einsetzen, das neben Attention auch Norm, Router, Codec oder Head verändert.

**Abnahme:** Kontextabhängige Attention-Zeit sinkt einschließlich Merge. Tests umfassen nichttriviale Q/K/V, GQA-Zuordnung, vergiftete zukünftige Zeilen, Chunk-Grenzen und Ring-Wrap. Änderungen der Arithmetik bestehen P7, bevor sie als neue Default-Performance gelten.

### P6 Verbleibendes Decode gezielt verkürzen

Nach P1 und P5 die verbleibenden Millisekunden messen und nur die größten Posten bearbeiten:

| Restposten | Konkreter Versuch | Grenze der Optimierung |
| --- | --- | --- |
| H128 | Warp-Shuffles statt wiederholter CTA-Barrieren; paarweise Gate/Up-Dispatch; fusionierte Ausgabe | Sidecars getrennt; keine redundante Eingabetransformation je N-Tile einführen, ohne sie mitzumessen. |
| QKV/O-GEMV | FP8-Bitdecoder oder kleine Lookup-Tabelle; Eingaben einmal dekodieren; mehrere unabhängige Akkumulatoren; Packed Loads | Alle FP8-Bitmuster inklusive Subnormals, Vorzeichen und ungültiger Codes prüfen. Neue Summenreihenfolge separat validieren. |
| BF16-Head | FP32-GEMV-Auslastung, Register-Prefetch und Cache-Verhalten messen | 453 MB pro Token bleiben ohne Formatänderung. Quantisierung des Heads ist eine gesonderte Qualitätsänderung. |
| Norm/Router/Combine | Sinnvolle Fusionen und weniger leere CTAs | Routing-Ties, Normalisierung und Top-8-Reihenfolge erhalten. |
| Host-/Graph-Lücken | Forward, Head, Auswahl und Device-Tokenübergabe in einen Decode-Graph aufnehmen; gepinnten Kontrollpuffer prüfen | Bestehende Graphs nicht als fehlende Funktion behandeln. Streaming, Sampling, EOS und Abbruch dürfen nicht zusätzliche Tokens konsumieren. |

Greedy und stochastisches Sampling getrennt messen. Ein Head/Argmax-Fusionspfad darf Logits, Penalties oder stochastische Auswahl nicht stillschweigend umgehen. Kernelaufrufe sind erst dann das nächste Ziel, wenn ihre Lücken im Trace einen relevanten Anteil ausmachen.

**Abnahme:** beide langen Decode-Profile erreichen die Referenz. Als Zwischenmarke nach den ersten Änderungen sind 35–45 Tok/s nützlich; das wäre Fortschritt, aber keine erfüllte Aufgabe.

### P7 Qualität und alternative Arithmetik

Bitgleiche Änderungen zuerst ausschöpfen. Falls Prefill oder Decode danach unter dem Ziel bleiben, die Zeitgrenze erklären und die folgenden Varianten einzeln bewerten:

1. FP32-Akkumulation ohne durchgehende Kompensation für ausgewählte GEMV-/Head-/Attention-Operatoren, mit expliziten Rundungsgrenzen.
2. DP4A für Matrixprodukte mit INT8-Aktivierungen und geeigneten Gewichtsblöcken. Das unterscheidet sich grundlegend vom exakten DP4A-Hash aus P1. EXL3-Dekodierung, Skalen, Fehler und gegebenenfalls Residualpfad müssen definiert sein. Kein implizites Umdeuten der K4-Bits als lineare Int4-Gewichte.
3. Q8-KV erst dann, wenn Bandbreite oder Scratch-Aufnahme es rechtfertigen. Der vorhandene `Q8TokenHead128F32ScaleV1` ist kein identischer Ersatz für llama.cpp-Q8_0. Die aktuelle Runtime lehnt diesen Weg bei den kleinen Speicherprofilen ab; `kv_q8.cuh` einfach einzubinden reicht nicht. Layout, Store, Attention, Aufnahme, Snapshot und Profilidentität müssen zusammen umgestellt werden.

**Numerischer Vertrag:** Bei rein mechanischen Änderungen Codec und Operatorausgaben bitgleich vergleichen. Bei neuem Math-Profil Fehlerverteilung und Top-1-/Top-k-Logits auf teacher-forced, identischen Tokenfolgen erfassen; dadurch vermischt sich numerischer Fehler nicht mit bereits divergierter Freigenerierung. Keine feste Tokenidentität über 1000 frei generierte Tokens als einziges Qualitätsmaß verwenden.

Ein festes, gehashtes Korpus für deutsche/englische Prosa, Code und lange Kontexte einfrieren. NLL gegen bisherigen SIMT-Pfad und eine vertrauenswürdige Modellreferenz messen; dazu ausführbare Python-/TypeScript-/SQL-Aufgaben mit festen Seeds und dem ursprünglichen Smokeset. Die bisherige funktionale Qualifikation ersetzt diese Modellqualitätsprüfung nicht.

Vorgeschlagene vorab festzulegende Schranken für alternative Arithmetik: obere Grenze des gepaarten 95%-Konfidenzintervalls der NLL-Verschlechterung höchstens 0,01 Nat/Token und höchstens ein Prozentpunkt Verlust in der Aufgaben-Erfolgsrate. Ein dafür zu kleines Aufgabenset liefert keine Freigabe; dann Stichprobe vergrößern. Grenzen samt Korpus vor dem Tuning festhalten und nicht nach einem schnellen Ergebnis lockern. Greedy-/Seed-Reproduzierbarkeit, Streaming, Cancellation und Restore sind zusätzlich harte Funktionskriterien.

Ein Profil, das nur durch unqualifizierte Präzisionsänderung die Zielzahl erreicht, zählt nicht als abgeschlossene Performance-Verbesserung.

## Speicher und realistische Leistungsgrenzen

Bei 8192 Kontext belegen die BF16-KV-Daten genau `(7×8192 + 21×1024) × 2048 = 154 MiB`. Die bisherige Messung meldet etwa 1029 MiB frei; nach der vorgeschriebenen Reserve bleiben rechnerisch nur rund 329 MiB zusätzlicher Spielraum. Das ist ein Planungswert, keine neue Aufnahmegarantie. Neue Buffer einschließlich Graph-/Library-Workspace müssen vor und nach Capture tatsächlich aufgenommen werden.

Batch-Aktivierungen, Experten-Scratch, Staging und Attention-Partialwerte je nach Lebensdauer wiederverwenden. Mit einem zusätzlichen Workspace-Ziel von höchstens etwa 200–250 MiB bei Chunk256 beginnen und danach exakt bilanzieren. Zeiger müssen für Graph-Replay stabil bleiben. Keine Allokation, Gewichtsübertragung oder vollständige Dekodierung im Decode-Hotpath.

Q8-KV im vorhandenen Format benötigt 1056 statt 2048 Bytes pro Position und Layer. Bei 8192 spart es rund 74,6 MiB KV, zuzüglich eines möglichen Effekts auf entsprechend umgebaute Snapshots. Es spart keine GiB und erklärt allein keinen Faktor 50 beim Prefill.

Aus den konkreten Modelldimensionen ergibt sich pro Decode-Schritt folgende grobe Nutzlast:

| Gewichtsdaten | Bytes bei einmaligem Lesen |
| --- | ---: |
| EXL3-Payload der acht aktiven Experten über 28 Layer | 693,6 MB |
| Q/K/V/O als FP8 | 594,5 MB |
| BF16-Head | 453,0 MB |
| Summe ohne Sidecars, Router, KV und Scratch | 1,741 GB |

Für 57 Tok/s sind dafür etwa 99 GB/s erforderlich; zusätzliche KV-/Scratchzugriffe und Dekodierarbeit kommen hinzu. Das ist kein gemessener DRAM-Verkehr: Caches, wiederholte Loads und Transaktionsgrößen verändern ihn. Auf der Zielkarte die effektive Bandbreite messen und damit eine Untergrenze berechnen. Diese Nutzlast schließt das Decode-Ziel nicht offensichtlich aus, beweist seine Erreichbarkeit aber auch nicht.

Prefill benötigt für Experten und QKV/O ungefähr 3,964 GFLOP je Token, wenn alle Multiplikationen direkt in FP32 erfolgen; Attention, H128, Codec und Norm kommen hinzu. 900 Tok/s entsprechen bereits rund 3,57 TFLOP/s allein für diese Matrixprodukte. Darum ist Batch256 zwar notwendig, aber kein hinreichender Beweis für 900 Tok/s. P3/P4 müssen die nachhaltige Rechenleistung und Wiederverwendung tatsächlich zeigen. Bei zu geringer FP32-Leistungsreserve wird P7 relevant.

## Regressionen und Abschluss

Relevante Formen sind Promptlängen 1/3/4/15/16/31/32/63/64/127/128/255/256/257, 1023/1024/1025, 2000 und 4000. Lange Tests nahe 8192 müssen Prompt plus Ausgabe innerhalb der Kapazität halten. Für Routing Grenzfälle mit leeren, einzelnen und stark überfüllten Expertengruppen verwenden. Snapshot-Anker vor, auf und innerhalb einer Chunk-Grenze sowie nach Ring-Wrap testen.

Nach jedem strukturellen Kernel-/Cache-Umbau die passenden CPU-Referenztests und `compute-sanitizer` memcheck/racecheck ausführen. Das vollständige Checkpoint-Smokeset läuft nach Integration eines Pakets. SM86 und SM120a weiterhin bauen; geänderte gemeinsam genutzte Kernel zusätzlich auf der jeweiligen Hardware ausführen, bevor eine dort unveränderte Performance behauptet wird. Die Änderungen möglichst auf den Pascal-Pfad begrenzen.

Sol liefert zum Abschluss:

- den implementierten SM61-Pfad mit nachvollziehbaren Einzelcommits und aktualisierten Math-/Execution-Profilen;
- Rohmessungen, Laufzeitanteile, Peak-VRAM und Qualitätsresultate pro endgültigem Profil;
- dieselbe Vier-Zeilen-Tabelle wie oben mit wiederholter llama.cpp-Referenz, Streuung und Phasenvergleichen;
- dokumentierte verbleibende Lücke, falls ein Mindestziel trotz der geprüften Varianten nicht erreicht wird. Keine Umdeklaration des Ziels auf einen kürzeren Kontext oder ein anderes Testformat.

**Erster konkreter Arbeitsauftrag für Sol:** P0 vervollständigen und den bereits isoliert geprüften Codec aus P1 integrieren. Danach anhand der neuen Kernelzeiten entscheiden, wie viel Decode-Arbeit verbleibt; anschließend P2–P4 als zusammenhängenden Prefill-Umbau ausführen. Das vorhandene Experiment ist der Startpunkt, nicht die fertige Engine-Optimierung.
